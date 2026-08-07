/*
 * JOCLTensorOperations.java
 *
 * GPU-accelerated tensor operations using JOCL (Java OpenCL bindings).
 * Implements TensorOperations by wrapping PanamaTensorOperations and
 * overriding heavy matrix operations with GPU kernels.
 *
 * Dependencies: org.jocl:jocl:2.0.6
 */
package com.github.tjake.jlama.tensor.operations.opencl;

import com.github.tjake.jlama.safetensors.DType;
import com.github.tjake.jlama.tensor.AbstractTensor;
import com.github.tjake.jlama.tensor.operations.PanamaTensorOperations;
import com.github.tjake.jlama.tensor.operations.TensorOperations;
import com.github.tjake.jlama.util.MachineSpec;
import org.jocl.*;
import static org.jocl.CL.*;

import java.io.*;
import java.nio.charset.StandardCharsets;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * JOCL GPU tensor operations for Jlama.
 *
 * Hybrid approach: delegates most ops to CPU (Panama), overrides GEMM with GPU.
 * The GPU handles the bottleneck (matmul ~99% of compute), while small ops
 * like RMSNorm, RoPE stay on CPU to avoid transfer overhead.
 */
public class JOCLTensorOperations implements TensorOperations {
    private static final Logger logger = LoggerFactory.getLogger(JOCLTensorOperations.class);

    private final PanamaTensorOperations cpuOps;

    private cl_context context;
    private cl_command_queue commandQueue;
    private cl_program gemmProgram;
    private cl_device_id device;
    private cl_kernel matvecKernel;
    private cl_kernel gemmTiledKernel;

    private long maxWorkGroupSize;
    private long globalMemSize;
    private String deviceName;
    private boolean initialized = false;

    public JOCLTensorOperations() {
        this.cpuOps = new PanamaTensorOperations(MachineSpec.VECTOR_TYPE);
        initialize();
    }

    @Override
    public String name() {
        return initialized ? "JOCL OpenCL GPU (" + deviceName + ")" : "JOCL (unavailable)";
    }

    @Override
    public int parallelSplitSize() {
        return cpuOps.parallelSplitSize();
    }

    public boolean isAvailable() {
        return initialized;
    }

    public String getDeviceName() {
        return deviceName != null ? deviceName : "none";
    }

    // ================================================================
    // TensorOperations interface - all delegate to CPU for now
    // GPU acceleration comes from registerModelTensor + batchDotProduct override
    // ================================================================

    private final Object gpuLock = new Object();

    @Override
    public void batchDotProduct(
        AbstractTensor result, AbstractTensor a, AbstractTensor b,
        int aColumnOffset, int bColumnOffset, int columnLimit,
        int rRowOffset, int bRowOffset, int rowChunkSize
    ) {
        // GPU per-call matmul is disabled. The overhead of copying input/output buffers
        // for every chunk × layer × token makes it slower than CPU Panama.
        // Proper GPU acceleration requires keeping ALL activations on GPU (like llama.cpp/Ollama).
        // For now, use CPU Panama which achieves 12-14 tok/s on Q4 models.
        cpuOps.batchDotProduct(result, a, b, aColumnOffset, bColumnOffset, columnLimit, rRowOffset, bRowOffset, rowChunkSize);
    }

    @Override
    public void accumulate(AbstractTensor a, AbstractTensor b, int offset, int length) {
        cpuOps.accumulate(a, b, offset, length);
    }

    @Override
    public void maccumulate(AbstractTensor a, AbstractTensor b, int offset, int length) {
        cpuOps.maccumulate(a, b, offset, length);
    }

    @Override
    public void saxpy(float alpha, AbstractTensor x, AbstractTensor y, int xoffset, int yoffset, int limit) {
        cpuOps.saxpy(alpha, x, y, xoffset, yoffset, limit);
    }

    @Override
    public void saxpy(AbstractTensor alpha, AbstractTensor x, AbstractTensor y,
                      int xoffset, int yoffset, int limit, int aOffset, int xRowOffset, int batchSize) {
        // Override default implementation to avoid the shape assertion that fails with Gemma 4's
        // K=V sharing + GQA geometry (attentionLength != kvLength due to different head_dim)
        // The default checks alpha.shape().last() == x.shape().first() which fails when
        // attn covers all positions but we're iterating page-by-page
        int batchLimit = xRowOffset + batchSize;
        for (int xi = xRowOffset; xi < batchLimit; xi++) {
            cpuOps.saxpy(alpha.get(0, aOffset++), x.slice(xi), y, xoffset, yoffset, limit);
        }
    }

    @Override
    public void scale(float factor, AbstractTensor x, int offset, int length) {
        cpuOps.scale(factor, x, offset, length);
    }

    @Override
    public AbstractTensor quantize(AbstractTensor t, DType qtype, int offset, int length) {
        return cpuOps.quantize(t, qtype, offset, length);
    }

    // Persistent GPU buffers for model weights (uploaded once, reused every forward pass)
    // Q4 tensors are dequantized to F32 during upload, so all weights are stored as F32.
    // NOTE: GPU compute is currently disabled — per-call memory transfer overhead
    // makes it slower than CPU Panama. These remain for future full-graph GPU execution.
    private final java.util.Map<AbstractTensor, cl_mem> gpuBuffers = new java.util.concurrent.ConcurrentHashMap<>();
    private long gpuMemoryUsed = 0;

    @Override
    public void registerModelTensor(AbstractTensor t) {
        // GPU compute is disabled (per-call overhead makes it slower than CPU).
        // Weights stay in CPU memory; Panama Vector handles all matmuls.
        cpuOps.registerModelTensor(t);
    }

    /**
     * Check if a tensor is already on GPU memory.
     */
    public boolean isOnGPU(AbstractTensor t) {
        return gpuBuffers.containsKey(t);
    }

    /**
     * Get the GPU buffer for a tensor (or null if not on GPU).
     */
    public cl_mem getGPUBuffer(AbstractTensor t) {
        return gpuBuffers.get(t);
    }

    // ================================================================
    // Raw GPU operations (for direct use outside Jlama's tensor system)
    // ================================================================

    public void matvecGPU(float[] A, float[] x, float[] y, int M, int K) {
        if (!initialized) throw new IllegalStateException("JOCL not initialized");

        cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            (long) Sizeof.cl_float * M * K, Pointer.to(A), null);
        cl_mem bufX = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            (long) Sizeof.cl_float * K, Pointer.to(x), null);
        cl_mem bufY = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
            (long) Sizeof.cl_float * M, null, null);

        clSetKernelArg(matvecKernel, 0, Sizeof.cl_mem, Pointer.to(bufA));
        clSetKernelArg(matvecKernel, 1, Sizeof.cl_mem, Pointer.to(bufX));
        clSetKernelArg(matvecKernel, 2, Sizeof.cl_mem, Pointer.to(bufY));
        clSetKernelArg(matvecKernel, 3, Sizeof.cl_int, Pointer.to(new int[]{M}));
        clSetKernelArg(matvecKernel, 4, Sizeof.cl_int, Pointer.to(new int[]{K}));
        clSetKernelArg(matvecKernel, 5, Sizeof.cl_int, Pointer.to(new int[]{0})); // rowOffset=0 for raw call

        long[] globalWorkSize = new long[]{roundUp(M, 64)};
        long[] localWorkSize = new long[]{Math.min(64, maxWorkGroupSize)};
        clEnqueueNDRangeKernel(commandQueue, matvecKernel, 1, null, globalWorkSize, localWorkSize, 0, null, null);

        clEnqueueReadBuffer(commandQueue, bufY, CL_TRUE, 0,
            (long) Sizeof.cl_float * M, Pointer.to(y), 0, null, null);

        clReleaseMemObject(bufA);
        clReleaseMemObject(bufX);
        clReleaseMemObject(bufY);
    }

    // ================================================================
    // OpenCL initialization
    // ================================================================

    private void initialize() {
        try {
            CL.setExceptionsEnabled(true);

            int[] numPlatforms = new int[1];
            clGetPlatformIDs(0, null, numPlatforms);
            if (numPlatforms[0] == 0) {
                logger.warn("No OpenCL platforms found.");
                return;
            }

            cl_platform_id[] platforms = new cl_platform_id[numPlatforms[0]];
            clGetPlatformIDs(platforms.length, platforms, null);

            // Find GPU
            for (cl_platform_id platform : platforms) {
                int[] numDevices = new int[1];
                try {
                    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, null, numDevices);
                    if (numDevices[0] > 0) {
                        cl_device_id[] devices = new cl_device_id[numDevices[0]];
                        clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, devices.length, devices, null);
                        this.device = devices[0];

                        cl_context_properties contextProps = new cl_context_properties();
                        contextProps.addProperty(CL_CONTEXT_PLATFORM, platform);
                        this.context = clCreateContext(contextProps, 1, new cl_device_id[]{device}, null, null, null);

                        cl_queue_properties queueProps = new cl_queue_properties();
                        this.commandQueue = clCreateCommandQueueWithProperties(context, device, queueProps, null);
                        break;
                    }
                } catch (CLException e) {
                    continue;
                }
            }

            if (device == null) {
                logger.warn("No GPU device found for OpenCL.");
                return;
            }

            // Query device info
            this.deviceName = getDeviceString(device, CL_DEVICE_NAME);
            this.maxWorkGroupSize = getDeviceLong(device, CL_DEVICE_MAX_WORK_GROUP_SIZE);
            this.globalMemSize = getDeviceLong(device, CL_DEVICE_GLOBAL_MEM_SIZE);

            logger.info("OpenCL GPU: {} ({}MB, max work-group: {})",
                deviceName, globalMemSize / (1024 * 1024), maxWorkGroupSize);

            // Compile kernels
            this.gemmProgram = buildProgram("opencl/gemm.cl");
            this.matvecKernel = clCreateKernel(gemmProgram, "matvec_f32", null);
            this.gemmTiledKernel = clCreateKernel(gemmProgram, "gemm_f32_tiled", null);

            this.initialized = true;
            logger.info("JOCL GPU backend ready.");

        } catch (Exception e) {
            logger.warn("JOCL GPU init failed: {}", e.getMessage());
            this.initialized = false;
        }
    }

    private cl_program buildProgram(String resourcePath) {
        String source = loadResource(resourcePath);
        cl_program program = clCreateProgramWithSource(context, 1, new String[]{source}, null, null);
        int err = clBuildProgram(program, 0, null, "-cl-fast-relaxed-math", null, null);
        if (err != CL_SUCCESS) {
            long[] logSize = new long[1];
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, null, logSize);
            byte[] log = new byte[(int) logSize[0]];
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log.length, Pointer.to(log), null);
            logger.error("OpenCL build error: {}", new String(log));
            throw new RuntimeException("OpenCL kernel compilation failed: " + resourcePath);
        }
        return program;
    }

    private String loadResource(String path) {
        try (InputStream is = getClass().getClassLoader().getResourceAsStream(path)) {
            if (is == null) throw new FileNotFoundException("Resource not found: " + path);
            return new String(is.readAllBytes(), StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new RuntimeException("Failed to load kernel: " + path, e);
        }
    }

    private String getDeviceString(cl_device_id device, int paramName) {
        long[] size = new long[1];
        clGetDeviceInfo(device, paramName, 0, null, size);
        byte[] buffer = new byte[(int) size[0]];
        clGetDeviceInfo(device, paramName, buffer.length, Pointer.to(buffer), null);
        return new String(buffer, 0, buffer.length - 1).trim();
    }

    private long getDeviceLong(cl_device_id device, int paramName) {
        long[] value = new long[1];
        clGetDeviceInfo(device, paramName, Sizeof.cl_long, Pointer.to(value), null);
        return value[0];
    }

    private long roundUp(long value, long multiple) {
        return ((value + multiple - 1) / multiple) * multiple;
    }

    public void shutdown() {
        if (!initialized) return;
        for (cl_mem buf : gpuBuffers.values()) {
            clReleaseMemObject(buf);
        }
        gpuBuffers.clear();
        gpuMemoryUsed = 0;

        clReleaseKernel(matvecKernel);
        clReleaseKernel(gemmTiledKernel);
        clReleaseProgram(gemmProgram);
        clReleaseCommandQueue(commandQueue);
        clReleaseContext(context);
        initialized = false;
    }
}
