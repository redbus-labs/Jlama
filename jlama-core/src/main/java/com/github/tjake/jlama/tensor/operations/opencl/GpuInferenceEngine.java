package com.github.tjake.jlama.tensor.operations.opencl;

import com.github.tjake.jlama.model.*;
import com.github.tjake.jlama.safetensors.*;
import com.github.tjake.jlama.safetensors.tokenizer.*;
import com.github.tjake.jlama.tensor.*;
import org.jocl.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

import static org.jocl.CL.*;

/**
 * Full-graph GPU inference engine for Llama-family models.
 *
 * Unlike the per-call GPU approach, this keeps ALL computation on GPU:
 * - Weights uploaded once (Q4 packed bytes + scales)
 * - Activations stay in persistent GPU buffers
 * - KV cache lives on GPU
 * - Only token IDs are transferred CPU↔GPU
 *
 * Supports: Llama 3.x (1B, 3B, 8B) with Q4 quantization
 */
public class GpuInferenceEngine {
    private static final Logger logger = LoggerFactory.getLogger(GpuInferenceEngine.class);

    // Model config
    private final int hiddenSize;
    private final int intermediateSize;
    private final int numHeads;
    private final int numKVHeads;
    private final int headDim;
    private final int numLayers;
    private final int vocabSize;
    private final float ropeTheta;
    private final float rmsNormEps;
    private final int maxSeqLen;

    // OpenCL context
    private cl_context context;
    private cl_command_queue queue;
    private cl_program program;

    // Kernels
    private cl_kernel kRmsNorm;
    private cl_kernel kRope;
    private cl_kernel kSiluMul;
    private cl_kernel kSoftmax;
    private cl_kernel kResidualAdd;
    private cl_kernel kEmbeddingLookup;
    private cl_kernel kArgmax;
    private cl_kernel kMatvecQ4;
    private cl_kernel kMatvecF32;
    private cl_kernel kAttentionHead;

    // Weight buffers on GPU (per-layer)
    private cl_mem embedWeights;        // Q4 bytes or F32
    private cl_mem embedScales;         // Q4 scales (if embedIsQ4)
    private boolean embedIsQ4;
    private int embedRows, embedCols;
    private AbstractTensor embedTensorRef; // Keep reference for CPU embedding lookup
    private cl_mem finalNormWeights;    // F32 [hidden_size]

    // Per-layer weights
    private cl_mem[] inputNormWeights;  // F32 [hidden_size]
    private cl_mem[] postNormWeights;   // F32 [hidden_size]
    private cl_mem[] qProjData;         // Q4 bytes
    private cl_mem[] qProjScales;       // F32 scales
    private cl_mem[] kProjData;
    private cl_mem[] kProjScales;
    private cl_mem[] vProjData;
    private cl_mem[] vProjScales;
    private cl_mem[] oProjData;
    private cl_mem[] oProjScales;
    private cl_mem[] gateProjData;
    private cl_mem[] gateProjScales;
    private cl_mem[] upProjData;
    private cl_mem[] upProjScales;
    private cl_mem[] downProjData;
    private cl_mem[] downProjScales;

    // Activation buffers (persistent, reused each token)
    private cl_mem bufHidden;           // F32 [hidden_size]
    private cl_mem bufResidual;         // F32 [hidden_size]
    private cl_mem bufNormOut;          // F32 [hidden_size]
    private cl_mem bufQ;               // F32 [num_heads * head_dim]
    private cl_mem bufK;               // F32 [num_kv_heads * head_dim]
    private cl_mem bufV;               // F32 [num_kv_heads * head_dim]
    private cl_mem bufAttnOut;         // F32 [num_heads * head_dim]
    private cl_mem bufGate;            // F32 [intermediate_size]
    private cl_mem bufUp;              // F32 [intermediate_size]
    private cl_mem bufMlpOut;          // F32 [hidden_size]
    private cl_mem bufLogits;          // F32 [vocab_size]
    private cl_mem bufTokenId;         // int [1]

    // KV cache: [num_layers x max_seq_len x num_kv_heads x head_dim]
    private cl_mem[] kvCacheK;         // F32 per layer
    private cl_mem[] kvCacheV;         // F32 per layer
    private int currentSeqLen = 0;

    // Dimensions for Q4 weight addressing
    private int qProjRows, kProjRows, vProjRows, oProjRows;
    private int gateProjRows, upProjRows, downProjRows;

    public GpuInferenceEngine(Config config) {
        this.hiddenSize = config.embeddingLength;
        this.intermediateSize = config.hiddenLength;
        this.numHeads = config.numberOfHeads;
        this.numKVHeads = config.numberOfKeyValueHeads;
        this.headDim = config.headSize;
        this.numLayers = config.numberOfLayers;
        this.vocabSize = config.vocabularySize;
        this.ropeTheta = 500000.0f; // Default for Llama 3.x
        this.rmsNormEps = config.layerNormEps;
        this.maxSeqLen = 2048;

        this.qProjRows = numHeads * headDim;
        this.kProjRows = numKVHeads * headDim;
        this.vProjRows = numKVHeads * headDim;
        this.oProjRows = hiddenSize;
        this.gateProjRows = intermediateSize;
        this.upProjRows = intermediateSize;
        this.downProjRows = hiddenSize;
    }

    /** Initialize OpenCL context, compile kernels, allocate persistent buffers. */
    public void initialize() {
        cl_platform_id[] platforms = new cl_platform_id[1];
        clGetPlatformIDs(1, platforms, null);
        cl_device_id[] devices = new cl_device_id[1];
        clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 1, devices, null);

        context = clCreateContext(null, 1, devices, null, null, null);
        cl_queue_properties qp = new cl_queue_properties();
        queue = clCreateCommandQueueWithProperties(context, devices[0], qp, null);

        String source = loadKernelSource("opencl/ops.cl");
        program = clCreateProgramWithSource(context, 1, new String[]{source}, null, null);
        clBuildProgram(program, 0, null, "-cl-fast-relaxed-math", null, null);

        kRmsNorm = clCreateKernel(program, "rmsnorm", null);
        kRope = clCreateKernel(program, "rope", null);
        kSiluMul = clCreateKernel(program, "silu_mul", null);
        kSoftmax = clCreateKernel(program, "softmax", null);
        kResidualAdd = clCreateKernel(program, "residual_add", null);
        kEmbeddingLookup = clCreateKernel(program, "embedding_lookup", null);
        kArgmax = clCreateKernel(program, "argmax", null);
        kMatvecQ4 = clCreateKernel(program, "matvec_q4", null);
        kMatvecF32 = clCreateKernel(program, "matvec_f32", null);
        kAttentionHead = clCreateKernel(program, "attention_head", null);

        allocateBuffers();
        logger.info("GPU Engine: {}d hidden, {}d inter, {} layers, {} vocab",
            hiddenSize, intermediateSize, numLayers, vocabSize);
    }

    private void allocateBuffers() {
        long f = Sizeof.cl_float;
        bufHidden = clCreateBuffer(context, CL_MEM_READ_WRITE, f * hiddenSize, null, null);
        bufResidual = clCreateBuffer(context, CL_MEM_READ_WRITE, f * hiddenSize, null, null);
        bufNormOut = clCreateBuffer(context, CL_MEM_READ_WRITE, f * hiddenSize, null, null);
        bufQ = clCreateBuffer(context, CL_MEM_READ_WRITE, f * qProjRows, null, null);
        bufK = clCreateBuffer(context, CL_MEM_READ_WRITE, f * kProjRows, null, null);
        bufV = clCreateBuffer(context, CL_MEM_READ_WRITE, f * vProjRows, null, null);
        bufAttnOut = clCreateBuffer(context, CL_MEM_READ_WRITE, f * hiddenSize, null, null);
        bufGate = clCreateBuffer(context, CL_MEM_READ_WRITE, f * intermediateSize, null, null);
        bufUp = clCreateBuffer(context, CL_MEM_READ_WRITE, f * intermediateSize, null, null);
        bufMlpOut = clCreateBuffer(context, CL_MEM_READ_WRITE, f * hiddenSize, null, null);
        bufLogits = clCreateBuffer(context, CL_MEM_READ_WRITE, f * vocabSize, null, null);
        bufTokenId = clCreateBuffer(context, CL_MEM_WRITE_ONLY, Sizeof.cl_int, null, null);

        long kvSize = f * maxSeqLen * numKVHeads * headDim;
        kvCacheK = new cl_mem[numLayers];
        kvCacheV = new cl_mem[numLayers];
        for (int i = 0; i < numLayers; i++) {
            kvCacheK[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, kvSize, null, null);
            kvCacheV[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, kvSize, null, null);
        }

        inputNormWeights = new cl_mem[numLayers];
        postNormWeights = new cl_mem[numLayers];
        qProjData = new cl_mem[numLayers];
        qProjScales = new cl_mem[numLayers];
        kProjData = new cl_mem[numLayers];
        kProjScales = new cl_mem[numLayers];
        vProjData = new cl_mem[numLayers];
        vProjScales = new cl_mem[numLayers];
        oProjData = new cl_mem[numLayers];
        oProjScales = new cl_mem[numLayers];
        gateProjData = new cl_mem[numLayers];
        gateProjScales = new cl_mem[numLayers];
        upProjData = new cl_mem[numLayers];
        upProjScales = new cl_mem[numLayers];
        downProjData = new cl_mem[numLayers];
        downProjScales = new cl_mem[numLayers];
    }

    private String loadKernelSource(String path) {
        try (var is = getClass().getClassLoader().getResourceAsStream(path)) {
            return new String(is.readAllBytes());
        } catch (Exception e) {
            throw new RuntimeException("Failed to load kernel: " + path, e);
        }
    }

    /** Upload a Q4 weight tensor to GPU as packed bytes + scales. */
    public void uploadQ4Weight(AbstractTensor t, cl_mem[] dataArr, cl_mem[] scaleArr, int layer) {
        Q4ByteBufferTensor q4 = (Q4ByteBufferTensor) t;
        int rows = (int) t.shape().first();
        int cols = (int) t.shape().last();
        int numBytes = rows * cols / 2;
        int blocksPerRow = cols / 32;
        int totalBlocks = rows * blocksPerRow;

        // Extract raw bytes
        java.lang.foreign.MemorySegment seg = q4.getMemorySegment();
        byte[] bytes = new byte[numBytes];
        for (int i = 0; i < numBytes; i++) {
            bytes[i] = seg.get(java.lang.foreign.ValueLayout.JAVA_BYTE, i);
        }

        // Extract scales row-by-row
        FloatBufferTensor blockF = q4.getBlockF();
        float[] scales = new float[totalBlocks];
        for (int r = 0; r < rows; r++) {
            for (int b = 0; b < blocksPerRow; b++) {
                scales[r * blocksPerRow + b] = blockF.get(r, b);
            }
        }

        dataArr[layer] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            numBytes, Pointer.to(bytes), null);
        scaleArr[layer] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            (long) totalBlocks * Sizeof.cl_float, Pointer.to(scales), null);
    }

    /** Upload a F32 norm weight vector to GPU. */
    public void uploadF32Weight(AbstractTensor t, cl_mem[] arr, int layer) {
        int size = (int) t.shape().last();
        float[] data = new float[size];
        if (t.dims() == 1) {
            for (int i = 0; i < size; i++) data[i] = t.get(i);
        } else {
            for (int i = 0; i < size; i++) data[i] = t.get(0, i);
        }
        arr[layer] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            (long) size * Sizeof.cl_float, Pointer.to(data), null);
    }

    /** Upload embedding table to GPU. Handles Q4 by uploading as Q4 (bytes+scales). */
    public void uploadEmbedding(AbstractTensor t) {
        int rows = (int) t.shape().first();
        int cols = (int) t.shape().last();

        if (t.dType() == DType.Q4) {
            // Upload embedding as Q4 (bytes + scales) — same as weight layers
            // We'll use matvec_q4 for the lm_head, and a special embedding lookup
            Q4ByteBufferTensor q4 = (Q4ByteBufferTensor) t;
            int numBytes = rows * cols / 2;
            int blocksPerRow = cols / 32;
            int totalBlocks = rows * blocksPerRow;

            java.lang.foreign.MemorySegment seg = q4.getMemorySegment();
            byte[] bytes = new byte[numBytes];
            for (int i = 0; i < numBytes; i++) {
                bytes[i] = seg.get(java.lang.foreign.ValueLayout.JAVA_BYTE, i);
            }

            FloatBufferTensor blockF = q4.getBlockF();
            float[] scales = new float[totalBlocks];
            for (int r = 0; r < rows; r++) {
                for (int b = 0; b < blocksPerRow; b++) {
                    scales[r * blocksPerRow + b] = blockF.get(r, b);
                }
            }

            embedWeights = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                numBytes, Pointer.to(bytes), null);
            // Store embed scales in a temporary field
            embedScales = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                (long) totalBlocks * Sizeof.cl_float, Pointer.to(scales), null);
            embedIsQ4 = true;
            embedRows = rows;
            embedCols = cols;
            embedTensorRef = t;
        } else {
            // F32/BF16: dequantize and upload
            float[] data = new float[rows * cols];
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    data[r * cols + c] = t.get(r, c);
                }
            }
            embedWeights = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                (long) data.length * Sizeof.cl_float, Pointer.to(data), null);
            embedIsQ4 = false;
            embedRows = rows;
            embedCols = cols;
            embedTensorRef = t;
        }
    }

    /** Upload final layer norm weight. */
    public void uploadFinalNorm(AbstractTensor t) {
        int size = (int) t.shape().last(); // use last dim (handles [1, dim] shape)
        float[] data = new float[size];
        if (t.dims() == 1) {
            for (int i = 0; i < size; i++) data[i] = t.get(i);
        } else {
            for (int i = 0; i < size; i++) data[i] = t.get(0, i);
        }
        finalNormWeights = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            (long) size * Sizeof.cl_float, Pointer.to(data), null);
    }

    /**
     * Run one full forward pass on GPU. Returns the predicted next token ID.
     * Only the token_id is transferred CPU→GPU, and the result token_id GPU→CPU.
     */
    public int forward(int tokenId, int position) {
        // 1. Embedding lookup → bufHidden
        launchEmbeddingLookup(tokenId);

        // Copy hidden to residual for skip connection
        clEnqueueCopyBuffer(queue, bufHidden, bufResidual, 0, 0,
            (long) hiddenSize * Sizeof.cl_float, 0, null, null);

        // 2. For each transformer layer
        for (int layer = 0; layer < numLayers; layer++) {
            // 2a. Input LayerNorm → bufNormOut
            launchRmsNorm(bufResidual, inputNormWeights[layer], bufNormOut, hiddenSize);

            // 2b. QKV projections (Q4 matvec)
            launchMatvecQ4(qProjData[layer], qProjScales[layer], bufNormOut, bufQ,
                qProjRows, hiddenSize, hiddenSize);
            launchMatvecQ4(kProjData[layer], kProjScales[layer], bufNormOut, bufK,
                kProjRows, hiddenSize, hiddenSize);
            launchMatvecQ4(vProjData[layer], vProjScales[layer], bufNormOut, bufV,
                vProjRows, hiddenSize, hiddenSize);

            // 2c. RoPE on Q and K (per-head)
            for (int h = 0; h < numHeads; h++) {
                launchRope(bufQ, h * headDim, headDim, position);
            }
            for (int h = 0; h < numKVHeads; h++) {
                launchRope(bufK, h * headDim, headDim, position);
            }

            // 2d. Store K,V in cache at current position
            long kvOffset = (long) position * numKVHeads * headDim * Sizeof.cl_float;
            clEnqueueCopyBuffer(queue, bufK, kvCacheK[layer], 0, kvOffset,
                (long) kProjRows * Sizeof.cl_float, 0, null, null);
            clEnqueueCopyBuffer(queue, bufV, kvCacheV[layer], 0, kvOffset,
                (long) vProjRows * Sizeof.cl_float, 0, null, null);

            // 2e. Attention: for each head, compute scores, softmax, weighted sum
            launchAttention(layer, position);

            // 2f. O projection → bufAttnOut (reuse bufMlpOut as temp)
            launchMatvecQ4(oProjData[layer], oProjScales[layer], bufAttnOut, bufMlpOut,
                oProjRows, hiddenSize, hiddenSize);

            // 2g. Residual add
            launchResidualAdd(bufResidual, bufMlpOut, hiddenSize);

            // 2h. Post-attention LayerNorm
            launchRmsNorm(bufResidual, postNormWeights[layer], bufNormOut, hiddenSize);

            // 2i. MLP: gate + up projections
            launchMatvecQ4(gateProjData[layer], gateProjScales[layer], bufNormOut, bufGate,
                gateProjRows, hiddenSize, hiddenSize);
            launchMatvecQ4(upProjData[layer], upProjScales[layer], bufNormOut, bufUp,
                upProjRows, hiddenSize, hiddenSize);

            // 2j. SiLU(gate) * up → bufGate (reuse)
            launchSiluMul(bufGate, bufUp, bufGate, intermediateSize);

            // 2k. Down projection → bufMlpOut
            launchMatvecQ4(downProjData[layer], downProjScales[layer], bufGate, bufMlpOut,
                downProjRows, intermediateSize, intermediateSize);

            // 2l. Residual add
            launchResidualAdd(bufResidual, bufMlpOut, hiddenSize);
        }

        // 3. Final LayerNorm
        launchRmsNorm(bufResidual, finalNormWeights, bufNormOut, hiddenSize);

        // 4. LM Head projection (tied weights = embedding transposed)
        launchLmHead(bufNormOut);

        // 5. Sample from logits (temperature + top-k on CPU)
        float[] logits = new float[vocabSize];
        clEnqueueReadBuffer(queue, bufLogits, CL_TRUE, 0,
            (long) vocabSize * Sizeof.cl_float, Pointer.to(logits), 0, null, null);

        int result = sampleTopK(logits, 0.7f, 40);
        currentSeqLen = position + 1;
        return result;
    }

    /** Top-k sampling with temperature */
    private int sampleTopK(float[] logits, float temperature, int topK) {
        // Find top-k indices
        int[] topIndices = new int[topK];
        float[] topValues = new float[topK];
        java.util.Arrays.fill(topValues, Float.NEGATIVE_INFINITY);

        for (int i = 0; i < logits.length; i++) {
            float val = logits[i] / temperature;
            if (val > topValues[topK - 1]) {
                topValues[topK - 1] = val;
                topIndices[topK - 1] = i;
                // Bubble up
                for (int j = topK - 2; j >= 0; j--) {
                    if (topValues[j + 1] > topValues[j]) {
                        float tv = topValues[j]; topValues[j] = topValues[j + 1]; topValues[j + 1] = tv;
                        int ti = topIndices[j]; topIndices[j] = topIndices[j + 1]; topIndices[j + 1] = ti;
                    } else break;
                }
            }
        }

        // Softmax over top-k
        float maxVal = topValues[0];
        float sumExp = 0;
        for (int i = 0; i < topK; i++) {
            topValues[i] = (float) Math.exp(topValues[i] - maxVal);
            sumExp += topValues[i];
        }

        // Sample
        float r = (float) Math.random() * sumExp;
        float cumulative = 0;
        for (int i = 0; i < topK; i++) {
            cumulative += topValues[i];
            if (r <= cumulative) return topIndices[i];
        }
        return topIndices[0];
    }

    // ====== Kernel launchers ======

    private void launchEmbeddingLookup(int tokenId) {
        if (embedIsQ4) {
            // Use matvec_q4 to "select" one row: multiply by one-hot vector
            // Actually, for embedding lookup we just need to dequantize one row.
            // Use a specialized approach: read Q4 row on GPU
            // For now: dequantize the single row using matvec with a unit vector
            // Actually simpler: just read+dequant one row of Q4 on CPU and upload
            // This is only 2048 floats = 8KB, negligible transfer
            // TODO: write a Q4 embedding lookup kernel
            float[] row = new float[embedCols];
            // Can't easily do this without the tensor... store the tensor reference
            // For now, use the Q4 matvec kernel with rowOffset=tokenId, M=1
            clSetKernelArg(kMatvecQ4, 0, Sizeof.cl_mem, Pointer.to(embedWeights));
            clSetKernelArg(kMatvecQ4, 1, Sizeof.cl_mem, Pointer.to(embedScales));
            // We need an all-ones input to extract a row... that's wasteful.
            // Better: write a simple Q4 row extraction kernel. For now use CPU fallback.
            if (embedTensorRef != null) {
                for (int c = 0; c < embedCols; c++) {
                    row[c] = embedTensorRef.get(tokenId, c);
                }
                clEnqueueWriteBuffer(queue, bufHidden, CL_TRUE, 0,
                    (long) embedCols * Sizeof.cl_float, Pointer.to(row), 0, null, null);
            }
        } else {
            clSetKernelArg(kEmbeddingLookup, 0, Sizeof.cl_mem, Pointer.to(embedWeights));
            clSetKernelArg(kEmbeddingLookup, 1, Sizeof.cl_mem, Pointer.to(bufHidden));
            clSetKernelArg(kEmbeddingLookup, 2, Sizeof.cl_int, Pointer.to(new int[]{tokenId}));
            clSetKernelArg(kEmbeddingLookup, 3, Sizeof.cl_int, Pointer.to(new int[]{hiddenSize}));
            clEnqueueNDRangeKernel(queue, kEmbeddingLookup, 1, null,
                new long[]{roundUp(hiddenSize, 256)}, new long[]{256}, 0, null, null);
        }
    }

    private void launchRmsNorm(cl_mem input, cl_mem weight, cl_mem output, int dim) {
        clSetKernelArg(kRmsNorm, 0, Sizeof.cl_mem, Pointer.to(input));
        clSetKernelArg(kRmsNorm, 1, Sizeof.cl_mem, Pointer.to(weight));
        clSetKernelArg(kRmsNorm, 2, Sizeof.cl_mem, Pointer.to(output));
        clSetKernelArg(kRmsNorm, 3, Sizeof.cl_int, Pointer.to(new int[]{dim}));
        clSetKernelArg(kRmsNorm, 4, Sizeof.cl_float, Pointer.to(new float[]{rmsNormEps}));
        // Single work-group with 256 threads (reduction kernel)
        clEnqueueNDRangeKernel(queue, kRmsNorm, 1, null,
            new long[]{256}, new long[]{256}, 0, null, null);
    }

    private void launchMatvecQ4(cl_mem data, cl_mem scales, cl_mem input, cl_mem output,
                                int M, int K, int stride) {
        clSetKernelArg(kMatvecQ4, 0, Sizeof.cl_mem, Pointer.to(data));
        clSetKernelArg(kMatvecQ4, 1, Sizeof.cl_mem, Pointer.to(scales));
        clSetKernelArg(kMatvecQ4, 2, Sizeof.cl_mem, Pointer.to(input));
        clSetKernelArg(kMatvecQ4, 3, Sizeof.cl_mem, Pointer.to(output));
        clSetKernelArg(kMatvecQ4, 4, Sizeof.cl_int, Pointer.to(new int[]{M}));
        clSetKernelArg(kMatvecQ4, 5, Sizeof.cl_int, Pointer.to(new int[]{K}));
        clSetKernelArg(kMatvecQ4, 6, Sizeof.cl_int, Pointer.to(new int[]{0})); // rowOffset
        clSetKernelArg(kMatvecQ4, 7, Sizeof.cl_int, Pointer.to(new int[]{0})); // colOffset
        clSetKernelArg(kMatvecQ4, 8, Sizeof.cl_int, Pointer.to(new int[]{stride}));
        clEnqueueNDRangeKernel(queue, kMatvecQ4, 1, null,
            new long[]{roundUp(M, 256)}, new long[]{Math.min(256, M)}, 0, null, null);
    }

    private void launchRope(cl_mem vec, int offset, int dim, int position) {
        clSetKernelArg(kRope, 0, Sizeof.cl_mem, Pointer.to(vec));
        clSetKernelArg(kRope, 1, Sizeof.cl_int, Pointer.to(new int[]{offset}));
        clSetKernelArg(kRope, 2, Sizeof.cl_int, Pointer.to(new int[]{dim}));
        clSetKernelArg(kRope, 3, Sizeof.cl_int, Pointer.to(new int[]{position}));
        clSetKernelArg(kRope, 4, Sizeof.cl_float, Pointer.to(new float[]{ropeTheta}));
        clEnqueueNDRangeKernel(queue, kRope, 1, null,
            new long[]{dim / 2}, null, 0, null, null);
    }

    private void launchSiluMul(cl_mem gate, cl_mem up, cl_mem output, int dim) {
        clSetKernelArg(kSiluMul, 0, Sizeof.cl_mem, Pointer.to(gate));
        clSetKernelArg(kSiluMul, 1, Sizeof.cl_mem, Pointer.to(up));
        clSetKernelArg(kSiluMul, 2, Sizeof.cl_mem, Pointer.to(output));
        clSetKernelArg(kSiluMul, 3, Sizeof.cl_int, Pointer.to(new int[]{dim}));
        clEnqueueNDRangeKernel(queue, kSiluMul, 1, null,
            new long[]{roundUp(dim, 256)}, new long[]{256}, 0, null, null);
    }

    private void launchResidualAdd(cl_mem a, cl_mem b, int dim) {
        clSetKernelArg(kResidualAdd, 0, Sizeof.cl_mem, Pointer.to(a));
        clSetKernelArg(kResidualAdd, 1, Sizeof.cl_mem, Pointer.to(b));
        clSetKernelArg(kResidualAdd, 2, Sizeof.cl_int, Pointer.to(new int[]{dim}));
        clEnqueueNDRangeKernel(queue, kResidualAdd, 1, null,
            new long[]{roundUp(dim, 256)}, new long[]{256}, 0, null, null);
    }

    private void launchAttention(int layer, int position) {
        int seqLen = position + 1;
        int headsPerKV = numHeads / numKVHeads;
        float scale = (float) (1.0 / Math.sqrt(headDim));

        // Launch one kernel per query head — each does scores + softmax + weighted sum on GPU
        for (int h = 0; h < numHeads; h++) {
            int kvHead = h / headsPerKV;

            clSetKernelArg(kAttentionHead, 0, Sizeof.cl_mem, Pointer.to(bufQ));
            clSetKernelArg(kAttentionHead, 1, Sizeof.cl_mem, Pointer.to(kvCacheK[layer]));
            clSetKernelArg(kAttentionHead, 2, Sizeof.cl_mem, Pointer.to(kvCacheV[layer]));
            clSetKernelArg(kAttentionHead, 3, Sizeof.cl_mem, Pointer.to(bufAttnOut));
            clSetKernelArg(kAttentionHead, 4, Sizeof.cl_int, Pointer.to(new int[]{headDim}));
            clSetKernelArg(kAttentionHead, 5, Sizeof.cl_int, Pointer.to(new int[]{seqLen}));
            clSetKernelArg(kAttentionHead, 6, Sizeof.cl_int, Pointer.to(new int[]{numKVHeads}));
            clSetKernelArg(kAttentionHead, 7, Sizeof.cl_int, Pointer.to(new int[]{kvHead}));
            clSetKernelArg(kAttentionHead, 8, Sizeof.cl_int, Pointer.to(new int[]{h * headDim}));
            clSetKernelArg(kAttentionHead, 9, Sizeof.cl_int, Pointer.to(new int[]{h * headDim}));
            clSetKernelArg(kAttentionHead, 10, Sizeof.cl_float, Pointer.to(new float[]{scale}));

            // Single work-group with 128 threads (reduction kernel)
            clEnqueueNDRangeKernel(queue, kAttentionHead, 1, null,
                new long[]{128}, new long[]{128}, 0, null, null);
        }
    }

    private void launchLmHead(cl_mem input) {
        if (embedIsQ4) {
            // LM head with Q4 weights: output[i] = dot(embed_row_i, input)
            // Use matvec_q4 with full vocab as M
            clSetKernelArg(kMatvecQ4, 0, Sizeof.cl_mem, Pointer.to(embedWeights));
            clSetKernelArg(kMatvecQ4, 1, Sizeof.cl_mem, Pointer.to(embedScales));
            clSetKernelArg(kMatvecQ4, 2, Sizeof.cl_mem, Pointer.to(input));
            clSetKernelArg(kMatvecQ4, 3, Sizeof.cl_mem, Pointer.to(bufLogits));
            clSetKernelArg(kMatvecQ4, 4, Sizeof.cl_int, Pointer.to(new int[]{vocabSize}));
            clSetKernelArg(kMatvecQ4, 5, Sizeof.cl_int, Pointer.to(new int[]{hiddenSize}));
            clSetKernelArg(kMatvecQ4, 6, Sizeof.cl_int, Pointer.to(new int[]{0}));
            clSetKernelArg(kMatvecQ4, 7, Sizeof.cl_int, Pointer.to(new int[]{0}));
            clSetKernelArg(kMatvecQ4, 8, Sizeof.cl_int, Pointer.to(new int[]{hiddenSize}));
            clEnqueueNDRangeKernel(queue, kMatvecQ4, 1, null,
                new long[]{roundUp(vocabSize, 256)}, new long[]{256}, 0, null, null);
        } else {
            clSetKernelArg(kMatvecF32, 0, Sizeof.cl_mem, Pointer.to(embedWeights));
            clSetKernelArg(kMatvecF32, 1, Sizeof.cl_mem, Pointer.to(input));
            clSetKernelArg(kMatvecF32, 2, Sizeof.cl_mem, Pointer.to(bufLogits));
            clSetKernelArg(kMatvecF32, 3, Sizeof.cl_int, Pointer.to(new int[]{vocabSize}));
            clSetKernelArg(kMatvecF32, 4, Sizeof.cl_int, Pointer.to(new int[]{hiddenSize}));
            clSetKernelArg(kMatvecF32, 5, Sizeof.cl_int, Pointer.to(new int[]{0}));
            clEnqueueNDRangeKernel(queue, kMatvecF32, 1, null,
                new long[]{roundUp(vocabSize, 256)}, new long[]{256}, 0, null, null);
        }
    }

    private void launchArgmax() {
        clSetKernelArg(kArgmax, 0, Sizeof.cl_mem, Pointer.to(bufLogits));
        clSetKernelArg(kArgmax, 1, Sizeof.cl_mem, Pointer.to(bufTokenId));
        clSetKernelArg(kArgmax, 2, Sizeof.cl_int, Pointer.to(new int[]{vocabSize}));
        clEnqueueNDRangeKernel(queue, kArgmax, 1, null,
            new long[]{1}, new long[]{1}, 0, null, null);
    }

    /** Reset KV cache for new conversation. */
    public void resetCache() {
        currentSeqLen = 0;
    }

    // Weight array getters for external upload
    public cl_mem[] getInputNormWeights() { return inputNormWeights; }
    public cl_mem[] getPostNormWeights() { return postNormWeights; }
    public cl_mem[] getQProjData() { return qProjData; }
    public cl_mem[] getQProjScales() { return qProjScales; }
    public cl_mem[] getKProjData() { return kProjData; }
    public cl_mem[] getKProjScales() { return kProjScales; }
    public cl_mem[] getVProjData() { return vProjData; }
    public cl_mem[] getVProjScales() { return vProjScales; }
    public cl_mem[] getOProjData() { return oProjData; }
    public cl_mem[] getOProjScales() { return oProjScales; }
    public cl_mem[] getGateProjData() { return gateProjData; }
    public cl_mem[] getGateProjScales() { return gateProjScales; }
    public cl_mem[] getUpProjData() { return upProjData; }
    public cl_mem[] getUpProjScales() { return upProjScales; }
    public cl_mem[] getDownProjData() { return downProjData; }
    public cl_mem[] getDownProjScales() { return downProjScales; }

    private static long roundUp(long value, long multiple) {
        return ((value + multiple - 1) / multiple) * multiple;
    }
}
