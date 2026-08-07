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
 * Q4 Matrix-vector multiply: y[row] = sum(dequant(A_q4[row, colOff:colOff+K]) * x[k])
 *
 * @param A_q4        Packed Q4 weight data (full tensor, flat row-major)
 * @param A_scales    Block scales (full tensor, flat row-major: [totalRows x tensorWidth/32])
 * @param x           Input vector [K] in F32
 * @param y           Output vector [M] in F32
 * @param M           Number of output rows to compute
 * @param K           Number of columns to dot-product (columnLimit)
 * @param rowOffset   Start row in weight matrix
 * @param colOffset   Column offset in weight tensor (bColumnOffset)
 * @param tensorWidth Full width of the weight tensor (total columns)
 */
__kernel void matvec_q4(
    __global const uchar* A_q4,      // packed nibbles (full tensor)
    __global const float* A_scales,   // block scales (full tensor)
    __global const float* x,          // input vector (F32)
    __global float* y,                // output vector (F32)
    const int M,
    const int K,
    const int rowOffset,
    const int colOffset,
    const int tensorWidth
) {
    const int row = get_global_id(0);
    if (row >= M) return;

    const int actualRow = row + rowOffset;
    const int bytesPerRow = tensorWidth / 2;       // full row in bytes
    const int blocksPerRow = tensorWidth / Q4_BLOCK_SIZE;  // full row in blocks

    // Starting block and position within the row
    const int startBlock = colOffset / Q4_BLOCK_SIZE;

    // Offset into packed data for this row
    const int rowDataOffset = actualRow * bytesPerRow;
    // Offset into scales for this row
    const int rowScaleOffset = actualRow * blocksPerRow;

    float sum = 0.0f;
    const int numBlocks = K / Q4_BLOCK_SIZE;

    for (int b = 0; b < numBlocks; b++) {
        int block = startBlock + b;
        float scale = A_scales[rowScaleOffset + block];
        int blockByteOffset = rowDataOffset + block * Q4_HALF_BLOCK;
        int inputStart = b * Q4_BLOCK_SIZE;

        // Process 32 weights per block (16 bytes, 2 nibbles each)
        for (int j = 0; j < Q4_HALF_BLOCK; j++) {
            uchar packed = A_q4[blockByteOffset + j];

            // Low nibble: position [j] in the block (first half)
            int x0 = (packed & 0x0F) - 8;
            float w0 = x0 * scale;
            sum += w0 * x[inputStart + j];

            // High nibble: position [j + 16] in the block (second half)
            int x1 = ((packed >> 4) & 0x0F) - 8;
            float w1 = x1 * scale;
            sum += w1 * x[inputStart + j + Q4_HALF_BLOCK];
        }
    }

    y[row] = sum;
}
