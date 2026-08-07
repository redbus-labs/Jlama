/**
 * ops.cl - GPU inference kernels for LLM transformer forward pass
 *
 * All activations are F32. Weights are Q4 (packed nibbles + float scales).
 * These kernels keep everything on GPU — no CPU round-trips between layers.
 */

// ============================================================
// RMS Norm: output[i] = (input[i] / rms) * weight[i]
// where rms = sqrt(mean(input^2) + eps)
// ============================================================
__kernel void rmsnorm(
    __global const float* input,    // [dim]
    __global const float* weight,   // [dim] - learnable scale
    __global float* output,         // [dim]
    const int dim,
    const float eps
) {
    // First pass: compute sum of squares (single work-group reduction)
    __local float partial_sums[256];
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);

    float local_sum = 0.0f;
    for (int i = lid; i < dim; i += lsize) {
        float val = input[i];
        local_sum += val * val;
    }
    partial_sums[lid] = local_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Tree reduction
    for (int s = lsize / 2; s > 0; s >>= 1) {
        if (lid < s) {
            partial_sums[lid] += partial_sums[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float rms = sqrt(partial_sums[0] / dim + eps);

    // Second pass: normalize and scale
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = lid; i < dim; i += lsize) {
        output[i] = (input[i] / rms) * weight[i];
    }
}

// ============================================================
// Rotary Position Embedding (RoPE) - HuggingFace/Llama format
// Pairs element [i] with element [i + dim/2] (halved, not interleaved)
// BATCHED version: processes all heads in one kernel launch
// ============================================================
__kernel void rope(
    __global float* vec,       // full Q or K buffer [numHeads * headDim]
    const int vec_offset,      // UNUSED (kept for compat) - always 0 for batched
    const int dim,             // head_dim (e.g., 64)
    const int position,        // token position in sequence
    const float theta_base     // 500000.0 for Llama 3.x
) {
    // global_id covers all pairs across all heads
    const int gid = get_global_id(0);
    const int half_dim = dim / 2;
    const int head = gid / half_dim;      // which head
    const int i = gid % half_dim;         // pair index within head

    int headOffset = head * dim;
    int idx0 = headOffset + i;            // first half
    int idx1 = headOffset + i + half_dim;  // second half

    float freq = 1.0f / pow(theta_base, (float)(2 * i) / (float)dim);
    float angle = position * freq;
    float cos_val = cos(angle);
    float sin_val = sin(angle);

    float x0 = vec[idx0];
    float x1 = vec[idx1];

    vec[idx0] = x0 * cos_val - x1 * sin_val;
    vec[idx1] = x0 * sin_val + x1 * cos_val;
}

// ============================================================
// SiLU (Swish) activation * gate: output = silu(gate) * up
// silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
// ============================================================
__kernel void silu_mul(
    __global const float* gate,    // [dim] - gate projection output
    __global const float* up,      // [dim] - up projection output
    __global float* output,        // [dim]
    const int dim
) {
    const int i = get_global_id(0);
    if (i >= dim) return;

    float g = gate[i];
    float silu_g = g / (1.0f + exp(-g));
    output[i] = silu_g * up[i];
}

// ============================================================
// Softmax: output[i] = exp(input[i] - max) / sum(exp(input - max))
// Single work-group for the attention score vector
// ============================================================
__kernel void softmax(
    __global float* data,      // [len] - in-place softmax
    const int len
) {
    __local float shared[256];
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);

    // Find max
    float local_max = -INFINITY;
    for (int i = lid; i < len; i += lsize) {
        local_max = fmax(local_max, data[i]);
    }
    shared[lid] = local_max;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = lsize / 2; s > 0; s >>= 1) {
        if (lid < s) shared[lid] = fmax(shared[lid], shared[lid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float max_val = shared[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Compute exp and sum
    float local_sum = 0.0f;
    for (int i = lid; i < len; i += lsize) {
        float e = exp(data[i] - max_val);
        data[i] = e;
        local_sum += e;
    }
    shared[lid] = local_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = lsize / 2; s > 0; s >>= 1) {
        if (lid < s) shared[lid] += shared[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float total_sum = shared[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Normalize
    for (int i = lid; i < len; i += lsize) {
        data[i] /= total_sum;
    }
}

// ============================================================
// Attention: compute scores, softmax, weighted sum for ONE head
// All on GPU — no CPU round-trips for KV cache reads
// ============================================================
__kernel void attention_head(
    __global const float* Q,        // [headDim] query for this head
    __global const float* K_cache,  // [maxSeqLen x numKVHeads x headDim] full KV cache
    __global const float* V_cache,  // same layout as K
    __global float* output,         // [headDim] attention output for this head
    const int headDim,
    const int seqLen,               // number of positions to attend to
    const int numKVHeads,
    const int kvHead,               // which KV head to use
    const int qHeadOffset,          // offset into Q buffer for this head
    const int outHeadOffset,        // offset into output buffer for this head
    const float scale               // 1/sqrt(headDim)
) {
    // Single work-group computes full attention for one head
    __local float scores[2048];     // max seq len supported
    __local float shared[256];

    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);

    // Step 1: Compute attention scores (Q dot K for each position)
    for (int pos = lid; pos < seqLen; pos += lsize) {
        float dot = 0.0f;
        int kOffset = pos * numKVHeads * headDim + kvHead * headDim;
        for (int d = 0; d < headDim; d++) {
            dot += Q[qHeadOffset + d] * K_cache[kOffset + d];
        }
        scores[pos] = dot * scale;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Step 2: Softmax - find max
    float local_max = -INFINITY;
    for (int i = lid; i < seqLen; i += lsize) {
        local_max = fmax(local_max, scores[i]);
    }
    shared[lid] = local_max;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = lsize / 2; s > 0; s >>= 1) {
        if (lid < s) shared[lid] = fmax(shared[lid], shared[lid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float max_val = shared[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Softmax - exp and sum
    float local_sum = 0.0f;
    for (int i = lid; i < seqLen; i += lsize) {
        float e = exp(scores[i] - max_val);
        scores[i] = e;
        local_sum += e;
    }
    shared[lid] = local_sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = lsize / 2; s > 0; s >>= 1) {
        if (lid < s) shared[lid] += shared[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float total_sum = shared[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Normalize
    for (int i = lid; i < seqLen; i += lsize) {
        scores[i] /= total_sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Step 3: Weighted sum of V
    for (int d = lid; d < headDim; d += lsize) {
        float sum = 0.0f;
        for (int pos = 0; pos < seqLen; pos++) {
            int vOffset = pos * numKVHeads * headDim + kvHead * headDim;
            sum += scores[pos] * V_cache[vOffset + d];
        }
        output[outHeadOffset + d] = sum;
    }
}

// ============================================================
// Residual Add: output[i] = a[i] + b[i]
// ============================================================
__kernel void residual_add(
    __global float* a,         // [dim] - accumulates in place
    __global const float* b,   // [dim]
    const int dim
) {
    const int i = get_global_id(0);
    if (i >= dim) return;
    a[i] += b[i];
}

// ============================================================
// Embedding Lookup: copy one row from embedding table
// Embedding stored as F32 (dequantized at upload) since it's small
// ============================================================
__kernel void embedding_lookup(
    __global const float* embed_table,  // [vocab_size x dim]
    __global float* output,             // [dim]
    const int token_id,
    const int dim
) {
    const int i = get_global_id(0);
    if (i >= dim) return;
    output[i] = embed_table[token_id * dim + i];
}

// ============================================================
// Argmax: find index of maximum value
// Returns result in output[0]
// ============================================================
__kernel void argmax(
    __global const float* data,    // [len]
    __global int* result,          // [1] - output token id
    const int len
) {
    // Single work-item version (simple, len = vocab_size ~32K-128K)
    // For production, use parallel reduction
    float max_val = -INFINITY;
    int max_idx = 0;
    for (int i = 0; i < len; i++) {
        if (data[i] > max_val) {
            max_val = data[i];
            max_idx = i;
        }
    }
    result[0] = max_idx;
}

// ============================================================
// Q4 Matrix-vector multiply (from gemm_q4.cl, included here for
// self-contained full-graph execution)
// ============================================================
#define Q4_BLOCK_SIZE 32
#define Q4_HALF_BLOCK 16

__kernel void matvec_q4(
    __global const uchar* A_q4,
    __global const float* A_scales,
    __global const float* x,
    __global float* y,
    const int M,
    const int K,
    const int rowOffset,
    const int colOffset,
    const int stride
) {
    const int row = get_global_id(0);
    if (row >= M) return;

    const int actualRow = row + rowOffset;
    const int bytesPerRow = stride / 2;
    const int blocksPerRow = stride / Q4_BLOCK_SIZE;
    const int startBlock = colOffset / Q4_BLOCK_SIZE;
    const int rowDataOffset = actualRow * bytesPerRow;
    const int rowScaleOffset = actualRow * blocksPerRow;
    const int numBlocks = K / Q4_BLOCK_SIZE;

    float sum = 0.0f;

    for (int b = 0; b < numBlocks; b++) {
        int block = startBlock + b;
        float scale = A_scales[rowScaleOffset + block];
        int blockByteOffset = rowDataOffset + block * Q4_HALF_BLOCK;
        int inputStart = b * Q4_BLOCK_SIZE;

        for (int j = 0; j < Q4_HALF_BLOCK; j++) {
            uchar packed = A_q4[blockByteOffset + j];

            int x0 = (packed & 0x0F) - 8;
            sum += (x0 * scale) * x[inputStart + j];

            int x1 = ((packed >> 4) & 0x0F) - 8;
            sum += (x1 * scale) * x[inputStart + j + Q4_HALF_BLOCK];
        }
    }

    y[row] = sum;
}

// ============================================================
// F32 Matrix-vector multiply (for embedding/lm_head which may be F32)
// ============================================================
__kernel void matvec_f32(
    __global const float* A,
    __global const float* x,
    __global float* y,
    const int M,
    const int K,
    const int rowOffset
) {
    const int row = get_global_id(0);
    if (row >= M) return;

    const int actualRow = row + rowOffset;
    const int offset = actualRow * K;

    float sum = 0.0f;
    int k = 0;
    for (; k + 3 < K; k += 4) {
        float4 a = vload4(0, A + offset + k);
        float4 b = vload4(0, x + k);
        sum += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    for (; k < K; k++) {
        sum += A[offset + k] * x[k];
    }

    y[row] = sum;
}
