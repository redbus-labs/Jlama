/*
 * Copyright 2024 T Jake Luciani
 *
 * The Jlama Project licenses this file to you under the Apache License,
 * version 2.0 (the "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at:
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tjake.jlama.model.gemma3;

import com.github.tjake.jlama.math.FloatConversions;
import com.github.tjake.jlama.model.*;
import com.github.tjake.jlama.model.functions.EmbedInput;
import com.github.tjake.jlama.model.functions.SampleOutput;
import com.github.tjake.jlama.model.llama.LlamaModel;
import com.github.tjake.jlama.safetensors.Config;
import com.github.tjake.jlama.safetensors.DType;
import com.github.tjake.jlama.safetensors.WeightLoader;
import com.github.tjake.jlama.safetensors.tokenizer.Tokenizer;
import com.github.tjake.jlama.tensor.AbstractTensor;
import com.github.tjake.jlama.tensor.operations.TensorOperationsProvider;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Optional;
import java.util.stream.IntStream;

/**
 * Gemma 3 model implementation.
 *
 * Key differences from Gemma 2:
 * - Hybrid attention: 5 sliding window (4096) + 1 global, repeating
 * - Weight prefix: model.language_model.layers.X (nested structure)
 * - Has query_pre_attn_scalar for attention scaling
 * - Separate v_proj (standard attention, no K=V sharing)
 * - 128K context with sliding window for efficiency
 *
 * Extends LlamaModel (same inheritance as Gemma2Model).
 */
public class Gemma3Model extends LlamaModel {
    private static final Logger logger = LoggerFactory.getLogger(Gemma3Model.class);

    private final float embeddingScalingFactor;
    private AbstractTensor wte;

    public Gemma3Model(
        Config config,
        WeightLoader weights,
        Tokenizer tokenizer,
        DType workingDType,
        DType workingQType,
        Optional<DType> modelQType
    ) {
        this(InferenceType.FULL_GENERATION, config, weights, tokenizer, workingDType, workingQType, modelQType);
    }

    public Gemma3Model(
        InferenceType inferenceType,
        Config config,
        WeightLoader weights,
        Tokenizer tokenizer,
        DType workingDType,
        DType workingQType,
        Optional<DType> modelQType
    ) {
        super(inferenceType, config, weights, tokenizer, workingDType, workingQType, modelQType);
        this.embeddingScalingFactor = FloatConversions.bFloat16ToFloat32(
            FloatConversions.float32ToBFloat16((float) Math.pow(c.embeddingLength, 0.5))
        );
    }

    @Override
    public ModelSupport.ModelType getModelType() {
        return ModelSupport.getModelType("GEMMA3");
    }

    @Override
    protected TransformerBlock[] loadTransformerBlockWeights() {
        DType qType = modelQType.orElse(this.modelDType);
        if (qType != this.modelDType) {
            logger.info("Quantizing model with {} - Please hold...", qType);
        }

        TransformerBlock[] transformerBlocks = new TransformerBlock[c.dctx().numberOfLayers];

        IntStream.range(c.dctx().layerStart, c.dctx().layerEnd).parallel().forEach(i -> {
            // Gemma 3 uses model.layers.X prefix (text-only) or model.language_model.layers.X (multimodal)
            // Try language_model prefix first, fall back to model.layers
            String base;
            try {
                weights.load("model.language_model.layers." + i + ".input_layernorm.weight");
                base = "model.language_model.layers." + i + ".";
            } catch (Exception e) {
                base = "model.layers." + i + ".";
            }

            String prefix = base + "self_attn.";

            // Gemma 3 has standard Q/K/V/O projections (NOT K=V like Gemma 4)
            CausalSelfAttention attention = new CausalSelfAttention(
                this,
                i,
                weights.load(prefix + "q_proj.weight", c.dctx(), true, false).quantize(qType),
                weights.load(prefix + "k_proj.weight", c.dctx(), true, false).quantize(qType),
                weights.load(prefix + "v_proj.weight", c.dctx(), true, false).quantize(qType),
                weights.load(prefix + "o_proj.weight", c.dctx(), false, true).quantize(qType)
            );

            // MLP: gate_proj, down_proj, up_proj (SwiGLU)
            prefix = base + "mlp.";
            MLPBlock mlp = new MLPBlock(
                this,
                c.activationFunction,
                weights.load(prefix + "gate_proj.weight", c.dctx(), true, false).quantize(qType),
                weights.load(prefix + "down_proj.weight", c.dctx(), false, true).quantize(qType),
                weights.load(prefix + "up_proj.weight", c.dctx(), true, false).quantize(qType)
            );

            // Gemma 3 has pre/post norms for attention and FFN (same as Gemma 2)
            transformerBlocks[i] = new TransformerBlock(
                this,
                i,
                new RMSNorm(this, weights.load(base + "input_layernorm.weight").quantize(qType), 1.0f),
                attention,
                new RMSNorm(this, weights.load(base + "post_attention_layernorm.weight").quantize(qType), 1.0f),
                new RMSNorm(this, weights.load(base + "pre_feedforward_layernorm.weight").quantize(qType), 1.0f),
                mlp,
                new RMSNorm(this, weights.load(base + "post_feedforward_layernorm.weight").quantize(qType), 1.0f)
            );

            Gemma3Config g3c = (Gemma3Config) c;
            if (g3c.isGlobalLayer(i)) {
                logger.debug("Layer {} = GLOBAL (full attention)", i);
            } else {
                logger.debug("Layer {} = LOCAL (sliding window: {})", i, g3c.getSlidingWindowSize());
            }
        });

        return transformerBlocks;
    }

    @Override
    protected EmbedInput loadInputWeights() {
        // Try language_model prefix first
        try {
            if (wte == null) wte = weights.load("model.language_model.embed_tokens.weight").quantize(workingDType);
        } catch (Exception e) {
            if (wte == null) wte = weights.load("model.embed_tokens.weight").quantize(workingDType);
        }

        return (inputToken, position) -> {
            AbstractTensor embedding = makeDenseTensor(c.embeddingLength);
            AbstractTensor at = wte.slice(true, inputToken);
            if (wte.dType() != embedding.dType()) {
                at = TensorOperationsProvider.get().quantize(at, embedding.dType(), 0, c.embeddingLength);
            }

            embedding.copyFrom(at, 0, 0, c.embeddingLength);

            // Gemma embedding scaling: sqrt(hidden_size)
            TensorOperationsProvider.get().scale(embeddingScalingFactor, embedding, 0, c.embeddingLength);

            return embedding;
        };
    }

    @Override
    protected SampleOutput loadOutputWeights() {
        DType qType = modelQType.orElse(this.modelDType);

        try {
            if (wte == null) wte = weights.load("model.language_model.embed_tokens.weight").quantize(workingDType);
        } catch (Exception e) {
            if (wte == null) wte = weights.load("model.embed_tokens.weight").quantize(workingDType);
        }

        // Try language_model prefix for norm
        AbstractTensor normWeight;
        try {
            normWeight = weights.load("model.language_model.norm.weight").quantize(qType);
        } catch (Exception e) {
            normWeight = weights.load("model.norm.weight").quantize(qType);
        }

        final LayerNorm layerNorm = new RMSNorm(this, normWeight, 1.0f);

        return new SampleOutput() {
            @Override
            public LayerNorm getOutputLayerNorm() {
                return layerNorm;
            }

            @Override
            public AbstractTensor getOutputLogitsWeights() {
                return wte;
            }
        };
    }
}
