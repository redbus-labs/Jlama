/**
 * ops.cl - Element-wise operations for LLM inference
 *
 * Contains: RMSNorm, Softmax, RoPE, SwiGLU, Scale, Accumulate
 */

// ============================================================
// RMS Normalization
// ============================================================

// Step 1: Compute sum of squares (reduction)
__kernel void rms_norm_ss(
    __global const float* input,   // [N]
    __global float* sum_sq,        // [1] output: sum of squares
    const int N
) {
    __local float localSums[256];
    int lid = get_local_id(0);
    int gid = get_global_id(0);

    float sum = 0.0f;
    for (int i = gid; i < N; i += get_global_size(0)) {
        float v = input[i];
        sum += v * v;
    }

    localSums[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Parallel reduction within work-group
    for (int stride = get_local_size(0) / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            localSums[lid] += localSums[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        atomic_add_float(sum_sq, localSums[0]);
    }
}

// Step 2: Apply normalization with weight
__kernel void rms_norm_apply(
    __global const float* input,   // [N]
    __global const float* weight,  // [N] learned scale
    __global float* output,        // [N]
    const float rms_inv,           // 1 / sqrt(mean_sq + eps)
    const int N
) {
    int i = get_global_id(0);
    if (i >= N) return;
    output[i] = input[i] * rms_inv * weight[i];
}

// Combined RMSNorm (single kernel, less efficient but simpler)
__kernel void rms_norm(
    __global const float* input,
    __global const float* weight,
    __global float* output,
    const int N,
    const float eps
) {
    // Compute sum of squares
    float ss = 0.0f;
    for (int i = 0; i < N; i++) {
        ss += input[i] * input[i];
    }
    float rms_inv = rsqrt(ss / (float)N + eps);

    // Apply normalization
    int i = get_global_id(0);
    if (i >= N) return;
    output[i] = input[i] * rms_inv * weight[i];
}

// ============================================================
// Softmax
// ============================================================

// Softmax over a vector (for attention scores)
// Note: For production, split into max-reduction, exp, sum-reduction, divide
__kernel void softmax(
    __global float* data,    // [N] in-place
    const int N
) {
    // Find max for numerical stability
    float maxVal = -INFINITY;
    for (int i = 0; i < N; i++) {
        maxVal = fmax(maxVal, data[i]);
    }

    // Compute exp and sum
    float sumExp = 0.0f;
    for (int i = 0; i < N; i++) {
        data[i] = exp(data[i] - maxVal);
        sumExp += data[i];
    }

    // Normalize
    float invSum = 1.0f / sumExp;
    for (int i = 0; i < N; i++) {
        data[i] *= invSum;
    }
}

// Parallel softmax: each work-item handles one row
__kernel void softmax_rows(
    __global float* data,    // [rows x cols]
    const int cols
) {
    int row = get_global_id(0);
    int offset = row * cols;

    float maxVal = -INFINITY;
    for (int i = 0; i < cols; i++) {
        maxVal = fmax(maxVal, data[offset + i]);
    }

    float sumExp = 0.0f;
    for (int i = 0; i < cols; i++) {
        data[offset + i] = exp(data[offset + i] - maxVal);
        sumExp += data[offset + i];
    }

    float invSum = 1.0f / sumExp;
    for (int i = 0; i < cols; i++) {
        data[offset + i] *= invSum;
    }
}

// ============================================================
// Rotary Position Embedding (RoPE)
// ============================================================

// Apply RoPE rotation to query/key vectors
__kernel void rope_apply(
    __global float* vec,         // [headSize] query or key vector
    __global const float* freqs_cos,  // [headSize/2] precomputed cos
    __global const float* freqs_sin,  // [headSize/2] precomputed sin
    const int headSize,
    const int offset              // head offset into vec
) {
    int i = get_global_id(0);
    int halfHead = headSize / 2;
    if (i >= halfHead) return;

    int idx0 = offset + i;
    int idx1 = offset + i + halfHead;

    float v0 = vec[idx0];
    float v1 = vec[idx1];
    float fc = freqs_cos[i];
    float fs = freqs_sin[i];

    // Complex rotation: (v0 + i*v1) * (cos + i*sin)
    vec[idx0] = v0 * fc - v1 * fs;
    vec[idx1] = v0 * fs + v1 * fc;
}

// ============================================================
// SwiGLU Activation (used in FFN/MLP)
// ============================================================

// SwiGLU: output = silu(gate) * up
// where silu(x) = x * sigmoid(x)
__kernel void swiglu(
    __global const float* gate,   // [N] from gate_proj
    __global const float* up,     // [N] from up_proj
    __global float* output,       // [N]
    const int N
) {
    int i = get_global_id(0);
    if (i >= N) return;

    float g = gate[i];
    float silu = g / (1.0f + exp(-g)); // silu = x * sigmoid(x)
    output[i] = silu * up[i];
}

// GELU activation (used in Gemma 4)
// gelu_pytorch_tanh approximation
__kernel void gelu_tanh(
    __global float* data,    // [N] in-place
    const int N
) {
    int i = get_global_id(0);
    if (i >= N) return;

    float x = data[i];
    // Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    float x3 = x * x * x;
    float inner = 0.7978845608f * (x + 0.044715f * x3); // sqrt(2/pi) ≈ 0.7978845608
    data[i] = 0.5f * x * (1.0f + tanh(inner));
}

// ============================================================
// Utility Kernels
// ============================================================

// Scale a vector: x *= scale
__kernel void scale(
    __global float* data,
    const float scale_factor,
    const int N
) {
    int i = get_global_id(0);
    if (i >= N) return;
    data[i] *= scale_factor;
}

// Accumulate: a += b
__kernel void accumulate(
    __global float* a,
    __global const float* b,
    const int N
) {
    int i = get_global_id(0);
    if (i >= N) return;
    a[i] += b[i];
}

// Clamp (for ClippableLinear)
__kernel void clamp_tensor(
    __global float* data,
    const float min_val,
    const float max_val,
    const int N
) {
    int i = get_global_id(0);
    if (i >= N) return;
    data[i] = fmax(min_val, fmin(max_val, data[i]));
}

// Copy with type conversion (BF16 -> F32)
__kernel void bf16_to_f32(
    __global const ushort* input,   // BF16 stored as ushort
    __global float* output,
    const int N
) {
    int i = get_global_id(0);
    if (i >= N) return;
    // BF16 to F32: shift left by 16 bits
    uint bits = (uint)input[i] << 16;
    output[i] = as_float(bits);
}

// Atomic float add helper (for reductions)
inline void atomic_add_float(__global float* addr, float val) {
    union { unsigned int u; float f; } old_val, new_val;
    do {
        old_val.f = *addr;
        new_val.f = old_val.f + val;
    } while (atomic_cmpxchg((__global unsigned int*)addr, old_val.u, new_val.u) != old_val.u);
}
