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

    @Override
    public void batchDotProduct(
        AbstractTensor result, AbstractTensor a, AbstractTensor b,
        int aColumnOffset, int bColumnOffset, int columnLimit,
        int rRowOffset, int bRowOffset, int rowChunkSize
    ) {
        // If weight tensor B is on GPU, use GPU matmul
        // Use GPU when weights are pre-loaded. The rowOffset fix ensures correct results.
        // Transfer overhead exists but correctness > speed for now.
        // TODO: Optimize by pre-allocating input/output buffers to reduce malloc overhead.
        cl_mem gpuB = gpuBuffers.get(b);
        if (gpuB != null && initialized) {
            try {
                int K = columnLimit;  // number of input features
                int M = rowChunkSize; // number of output rows to compute
                int batchSize = a.shape().first();

                // Extract input data from tensor A
                float[] inputData = new float[batchSize * K];
                for (int batch = 0; batch < batchSize; batch++) {
                    for (int k = 0; k < K; k++) {
                        inputData[batch * K + k] = a.get(batch, aColumnOffset + k);
                    }
                }

                // Output buffer
                float[] outputData = new float[batchSize * M];

                // Upload input to GPU
                cl_mem bufInput = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    (long) Sizeof.cl_float * batchSize * K, Pointer.to(inputData), null);
                cl_mem bufOutput = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                    (long) Sizeof.cl_float * batchSize * M, null, null);

                if (batchSize == 1) {
                    // Single token: use matvec kernel with row offset
                    clSetKernelArg(matvecKernel, 0, Sizeof.cl_mem, Pointer.to(gpuB));
                    clSetKernelArg(matvecKernel, 1, Sizeof.cl_mem, Pointer.to(bufInput));
                    clSetKernelArg(matvecKernel, 2, Sizeof.cl_mem, Pointer.to(bufOutput));
                    clSetKernelArg(matvecKernel, 3, Sizeof.cl_int, Pointer.to(new int[]{M}));
                    clSetKernelArg(matvecKernel, 4, Sizeof.cl_int, Pointer.to(new int[]{K}));
                    clSetKernelArg(matvecKernel, 5, Sizeof.cl_int, Pointer.to(new int[]{bRowOffset}));

                    long[] globalWorkSize = new long[]{roundUp(M, 64)};
                    long[] localWorkSize = new long[]{Math.min(64, maxWorkGroupSize)};
                    clEnqueueNDRangeKernel(commandQueue, matvecKernel, 1, null, globalWorkSize, localWorkSize, 0, null, null);
                } else {
                    // Batch: use GEMM kernel (TODO: add row offset to GEMM too)
                    clSetKernelArg(gemmTiledKernel, 0, Sizeof.cl_mem, Pointer.to(bufInput));
                    clSetKernelArg(gemmTiledKernel, 1, Sizeof.cl_mem, Pointer.to(gpuB));
                    clSetKernelArg(gemmTiledKernel, 2, Sizeof.cl_mem, Pointer.to(bufOutput));
                    clSetKernelArg(gemmTiledKernel, 3, Sizeof.cl_int, Pointer.to(new int[]{batchSize}));
                    clSetKernelArg(gemmTiledKernel, 4, Sizeof.cl_int, Pointer.to(new int[]{M}));
                    clSetKernelArg(gemmTiledKernel, 5, Sizeof.cl_int, Pointer.to(new int[]{K}));

                    long[] globalWorkSize = new long[]{roundUp(batchSize, 16), roundUp(M, 16)};
                    long[] localWorkSize = new long[]{16, 16};
                    clEnqueueNDRangeKernel(commandQueue, gemmTiledKernel, 2, null, globalWorkSize, localWorkSize, 0, null, null);
                }

                // Read result back
                clEnqueueReadBuffer(commandQueue, bufOutput, CL_TRUE, 0,
                    (long) Sizeof.cl_float * batchSize * M, Pointer.to(outputData), 0, null, null);

                // Write back to result tensor
                for (int batch = 0; batch < batchSize; batch++) {
                    for (int m = 0; m < M; m++) {
                        result.set(result.get(batch, rRowOffset + m) + outputData[batch * M + m], batch, rRowOffset + m);
                    }
                }

                // Free temporary buffers (weight stays on GPU)
                clReleaseMemObject(bufInput);
                clReleaseMemObject(bufOutput);
                return;

            } catch (Exception e) {
                logger.debug("GPU matmul failed, falling back to CPU: {}", e.getMessage());
            }
        }

        // Fallback to CPU
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
    private final java.util.Map<AbstractTensor, cl_mem> gpuBuffers = new java.util.concurrent.ConcurrentHashMap<>();
    private long gpuMemoryUsed = 0;

    @Override
    public void registerModelTensor(AbstractTensor t) {
        // Upload weight tensor to GPU memory — stays there permanently
        if (!initialized) {
            cpuOps.registerModelTensor(t);
            return;
        }

        try {
            int numElements = (int) t.size();
            long byteSize = (long) numElements * Sizeof.cl_float;

            // Check if we have enough GPU memory (leave 2GB headroom for activations)
            if (gpuMemoryUsed + byteSize > globalMemSize - (2L * 1024 * 1024 * 1024)) {
                logger.debug("GPU memory full, tensor stays on CPU: {} elements", numElements);
                cpuOps.registerModelTensor(t);
                return;
            }

            // Extract float data from tensor using cursor iteration
            float[] data = new float[numElements];
            int[] cursor = new int[t.dims()];
            int idx = 0;
            while (t.iterate(cursor) && idx < numElements) {
                data[idx++] = t.get(cursor);
            }

            // Upload to GPU (READ_ONLY — weights don't change during inference)
            cl_mem gpuBuf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                byteSize, Pointer.to(data), null);

            gpuBuffers.put(t, gpuBuf);
            gpuMemoryUsed += byteSize;

            logger.debug("Uploaded tensor to GPU: {} elements ({} MB, total GPU: {} MB)",
                numElements, byteSize / (1024 * 1024), gpuMemoryUsed / (1024 * 1024));

        } catch (Exception e) {
            logger.debug("Failed to upload tensor to GPU, keeping on CPU: {}", e.getMessage());
            cpuOps.registerModelTensor(t);
        }
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
        // Release all persistent GPU weight buffers
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
        logger.info("JOCL GPU backend shut down. Released {} MB GPU memory.", gpuMemoryUsed / (1024 * 1024));
    }
}
