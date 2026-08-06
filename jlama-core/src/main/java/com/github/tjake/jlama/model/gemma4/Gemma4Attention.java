/*
 * Gemma4Attention.java
 *
 * Custom attention for Gemma 4 that applies QK normalization (q_norm, k_norm)
 * after Q/K projections and before RoPE. This is required for correct attention
 * scores in the Gemma 4 architecture.
 *
 * QK Norm: RMSNorm applied per-head to Q and K vectors before RoPE rotation.
 * Without this, attention scores are essentially random noise.
 */
package com.github.tjake.jlama.model.gemma4;

import com.github.tjake.jlama.model.*;
import com.github.tjake.jlama.safetensors.Config;
import com.github.tjake.jlama.tensor.AbstractTensor;

/**
 * Gemma 4 attention with QK normalization.
 *
 * Extends CausalSelfAttention by adding RMSNorm to Q and K vectors
 * per-head before they enter the RoPE and attention computation.
 *
 * Flow: input → Q/K projection → QK Norm → RoPE → Attention
 */
public class Gemma4Attention extends CausalSelfAttention {

    private final AbstractTensor qNormWeights; // [kvLength] per-head norm for queries
    private final AbstractTensor kNormWeights; // [kvLength] per-head norm for keys
    private final float normEps;
    private final int headSize;
    private final int numHeads;
    private final int numKVHeads;

    public Gemma4Attention(
        AbstractModel m,
        int layerIndex,
        AbstractTensor queryAttnWeights,
        AbstractTensor keyAttnWeights,
        AbstractTensor valueAttnWeights,
        AbstractTensor outputProjectionWeights,
        AbstractTensor qNormWeights,
        AbstractTensor kNormWeights,
        float normEps
    ) {
        super(m, layerIndex, queryAttnWeights, keyAttnWeights, valueAttnWeights, outputProjectionWeights);
        this.qNormWeights = qNormWeights;
        this.kNormWeights = kNormWeights;
        this.normEps = normEps;
        // Get config values from the model's config via public accessors
        Config config = m.getConfig();
        this.headSize = config.headSize;
        this.numHeads = config.numberOfHeads;
        this.numKVHeads = config.numberOfKeyValueHeads;
    }

    /**
     * Apply RMSNorm to a single head's worth of data within a tensor.
     * Normalizes in-place: tensor[0, offset..offset+headSize]
     *
     * RMSNorm(x) = x * (1/rms) * weight
     * where rms = sqrt(mean(x^2) + eps)
     */
    public static void applyHeadNorm(AbstractTensor tensor, int offset, int headSize, AbstractTensor normWeight, int weightOffset, float eps) {
        // Compute RMS for this head slice
        float sumSq = 0.0f;
        for (int i = 0; i < headSize; i++) {
            float v = tensor.get(0, offset + i);
            sumSq += v * v;
        }
        float rmsInv = (float) (1.0 / Math.sqrt(sumSq / headSize + eps));

        // Apply norm: x = x * rmsInv * weight
        // normWeight may be 1D [totalSize] or 2D [1, totalSize]
        for (int i = 0; i < headSize; i++) {
            float v = tensor.get(0, offset + i);
            float w;
            if (normWeight.dims() == 1) {
                w = normWeight.get(weightOffset + i);
            } else {
                w = normWeight.get(0, weightOffset + i);
            }
            tensor.set(v * rmsInv * w, 0, offset + i);
        }
    }

    /**
     * Apply QK normalization to query and key tensors.
     * Called after projection, before RoPE.
     *
     * q_norm.weight shape: [numHeads * headSize]
     * k_norm.weight shape: [numKVHeads * headSize]
     */
    public void applyQKNorm(AbstractTensor query, AbstractTensor key) {
        // Get actual sizes from the norm weight tensors to avoid OOB
        int qNormSize = (int) qNormWeights.size();
        int kNormSize = (int) kNormWeights.size();

        // Normalize each query head (only if norm weights are large enough)
        int queryHeads = Math.min(numHeads, qNormSize / headSize);
        for (int h = 0; h < queryHeads; h++) {
            int offset = h * headSize;
            if (offset + headSize <= query.shape().last()) {
                applyHeadNorm(query, offset, headSize, qNormWeights, h * headSize, normEps);
            }
        }

        // Normalize each key head
        int keyHeads = Math.min(numKVHeads, kNormSize / headSize);
        for (int h = 0; h < keyHeads; h++) {
            int offset = h * headSize;
            if (offset + headSize <= key.shape().last()) {
                applyHeadNorm(key, offset, headSize, kNormWeights, h * headSize, normEps);
            }
        }
    }

    @Override
    protected void postProjectionHook(AbstractTensor queryBatch, AbstractTensor keyBatch, AbstractTensor valueBatch) {
        // Apply QK normalization per batch item
        int batchSize = queryBatch.shape().first();
        for (int bi = 0; bi < batchSize; bi++) {
            AbstractTensor query = queryBatch.slice(bi);
            AbstractTensor key = keyBatch.slice(bi);
            applyQKNorm(query, key);
        }
    }
}
