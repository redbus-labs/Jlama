/**
 * gemm_q4.cl - Matrix-vector multiply for Jlama's Q4 quantized format
 *
 * Jlama Q4 format:
 * - Block size: 32 weights
 * - Each block: 16 bytes (packed nibbles) + 1 float (scale)
 * - Packing: byte[i] contains positions [i] (low nibble) and [i+16] (high nibble)
 * - Dequant: value = (nibble - 8) * scale
 *
 * This kernel dequantizes and computes dot product in one pass,
 * avoiding the need to dequantize the entire weight matrix first.
 */

#define Q4_BLOCK_SIZE 32
#define Q4_HALF_BLOCK 16

/**
 * Q4 Matrix-vector multiply: y[row] = sum(dequant(A_q4[row]) * x[k])
 *
 * @param A_q4       Packed Q4 weight data [totalRows x K/2 bytes]
 * @param A_scales   Block scales [totalRows x K/32 floats]
 * @param x          Input vector [K] in F32
 * @param y          Output vector [M] in F32
 * @param M          Number of output rows to compute
 * @param K          Number of input features (original unpacked dimension)
 * @param rowOffset  Start row in weight matrix
 */
__kernel void matvec_q4(
    __global const uchar* A_q4,      // packed nibbles
    __global const float* A_scales,   // block scales
    __global const float* x,          // input vector (F32)
    __global float* y,                // output vector (F32)
    const int M,
    const int K,
    const int rowOffset
) {
    const int row = get_global_id(0);
    if (row >= M) return;

    const int actualRow = row + rowOffset;
    const int blocksPerRow = K / Q4_BLOCK_SIZE;
    const int bytesPerRow = K / 2;  // 2 values per byte

    // Offset into packed data for this row
    const int dataOffset = actualRow * bytesPerRow;
    // Offset into scales for this row
    const int scaleOffset = actualRow * blocksPerRow;

    float sum = 0.0f;

    for (int block = 0; block < blocksPerRow; block++) {
        float scale = A_scales[scaleOffset + block];
        int blockByteOffset = dataOffset + block * Q4_HALF_BLOCK;
        int blockStart = block * Q4_BLOCK_SIZE;

        // Process 32 weights per block (16 bytes, 2 nibbles each)
        for (int j = 0; j < Q4_HALF_BLOCK; j++) {
            uchar packed = A_q4[blockByteOffset + j];

            // Low nibble: position [blockStart + j]
            int x0 = (packed & 0x0F) - 8;
            float w0 = x0 * scale;
            sum += w0 * x[blockStart + j];

            // High nibble: position [blockStart + j + 16]
            int x1 = ((packed >> 4) & 0x0F) - 8;
            float w1 = x1 * scale;
            sum += w1 * x[blockStart + j + Q4_HALF_BLOCK];
        }
    }

    y[row] = sum;
}
