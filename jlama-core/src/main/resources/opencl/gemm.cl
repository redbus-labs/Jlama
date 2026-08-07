/**
 * gemm.cl - General Matrix Multiplication kernel for LLM inference
 *
 * Computes: C = A * B^T (transposed B, as used in linear projections)
 * Where A is [M x K] and B is [N x K] (stored row-major, used as transposed)
 *
 * Optimizations:
 * - Tiled computation with local memory (work-group level blocking)
 * - Vectorized loads (float4) where possible
 * - Coalesced global memory access
 *
 * Used for: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
 */

#define TILE_SIZE 16

// Standard GEMM: C[M,N] = A[M,K] * B[N,K]^T
// Each work-item computes one element of C
__kernel void gemm_f32(
    __global const float* A,    // [M x K]
    __global const float* B,    // [N x K] (transposed access)
    __global float* C,          // [M x N]
    const int M,
    const int N,
    const int K
) {
    const int row = get_global_id(0); // M dimension
    const int col = get_global_id(1); // N dimension

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += A[row * K + k] * B[col * K + k]; // B transposed
    }
    C[row * N + col] = sum;
}

// Tiled GEMM with local memory for better cache utilization
__kernel void gemm_f32_tiled(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int M,
    const int N,
    const int K
) {
    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

    const int row = get_group_id(0) * TILE_SIZE + get_local_id(0);
    const int col = get_group_id(1) * TILE_SIZE + get_local_id(1);
    const int localRow = get_local_id(0);
    const int localCol = get_local_id(1);

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tile of A into local memory
        int aCol = t * TILE_SIZE + localCol;
        if (row < M && aCol < K)
            tileA[localRow][localCol] = A[row * K + aCol];
        else
            tileA[localRow][localCol] = 0.0f;

        // Load tile of B (transposed) into local memory
        int bCol = t * TILE_SIZE + localRow;
        if (col < N && bCol < K)
            tileB[localRow][localCol] = B[col * K + bCol];
        else
            tileB[localRow][localCol] = 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute partial dot product for this tile
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[localRow][k] * tileB[k][localCol];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Dot product: result = sum(A[i] * B[i]) for a single vector pair
// Used for attention score computation
__kernel void dot_product_batch(
    __global const float* queries,   // [numHeads x headSize]
    __global const float* keys,      // [seqLen x kvHeads x headSize]
    __global float* scores,          // [numHeads x seqLen]
    const int numHeads,
    const int kvHeads,
    const int headSize,
    const int seqLen
) {
    const int head = get_global_id(0);
    const int pos = get_global_id(1);

    if (head >= numHeads || pos >= seqLen) return;

    int kvHead = head / (numHeads / kvHeads); // GQA mapping
    float sum = 0.0f;

    int qOffset = head * headSize;
    int kOffset = pos * kvHeads * headSize + kvHead * headSize;

    for (int i = 0; i < headSize; i++) {
        sum += queries[qOffset + i] * keys[kOffset + i];
    }

    scores[head * seqLen + pos] = sum;
}

// Matrix-vector multiply: y = A[rowOffset:rowOffset+M, colOffset:colOffset+K] * x
// Used for single-token inference (batch_size=1)
// rowOffset allows computing a slice of the full weight matrix
// colOffset allows starting from a column offset
// stride is the full row width of A in memory
__kernel void matvec_f32(
    __global const float* A,    // [totalRows x stride] weight matrix (full)
    __global const float* x,    // [K] input vector
    __global float* y,          // [M] output vector
    const int M,                // number of rows to compute
    const int K,                // input dimension (columns to process)
    const int rowOffset,        // start row in A
    const int colOffset,        // start column in A
    const int stride            // full row width in A
) {
    const int row = get_global_id(0);
    if (row >= M) return;

    float sum = 0.0f;
    int actualRow = row + rowOffset;
    int offset = actualRow * stride + colOffset;

    // Vectorized accumulation (process 4 elements at a time)
    int k = 0;
    for (; k + 3 < K; k += 4) {
        float4 a = vload4(0, A + offset + k);
        float4 b = vload4(0, x + k);
        sum += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    // Handle remainder
    for (; k < K; k++) {
        sum += A[offset + k] * x[k];
    }

    y[row] = sum;
}
