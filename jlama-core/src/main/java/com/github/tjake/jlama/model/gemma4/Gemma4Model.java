/*
 * Copyright 2024 T Jake Luciani
 *
 * The Jlama Project licenses this file to you under the Apache License,
 * version 2.0 (the "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at:
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */
package com.github.tjake.jlama.model.gemma4;

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
 * Gemma 4 Unified model implementation.
 *
 * Key differences from Gemma 2:
 * - Hybrid attention: alternating local (sliding window) and global (full) layers
 * - ClippableLinear: optional input/output clamping for numerical stability
 * - 256K context window via proportional RoPE on global layers
 * - Encoder-free: all modalities processed in single decoder (text-only here)
 * - 262K vocabulary with new chat tokens (<|turn>, <turn|>, etc.)
 *
 * Extends LlamaModel (same inheritance as Gemma2Model).
 */
public class Gemma4Model extends LlamaModel {
    private static final Logger logger = LoggerFactory.getLogger(Gemma4Model.class);

    private final float embeddingScalingFactor;
    private final Gemma4Config gemma4Config;
    private AbstractTensor wte;

    public Gemma4Model(
        Config config,
        WeightLoader weights,
        Tokenizer tokenizer,
        DType workingDType,
        DType workingQType,
        Optional<DType> modelQType
    ) {
        this(InferenceType.FULL_GENERATION, config, weights, tokenizer, workingDType, workingQType, modelQType);
    }

    public Gemma4Model(
        InferenceType inferenceType,
        Config config,
        WeightLoader weights,
        Tokenizer tokenizer,
        DType workingDType,
        DType workingQType,
        Optional<DType> modelQType
    ) {
        super(inferenceType, config, weights, tokenizer, workingDType, workingQType, modelQType);
        this.gemma4Config = (Gemma4Config) config;
        // Embedding scaling: sqrt(hidden_size), rounded to bf16 like Gemma 2
        this.embeddingScalingFactor = FloatConversions.bFloat16ToFloat32(
            FloatConversions.float32ToBFloat16((float) Math.pow(c.embeddingLength, 0.5))
        );
    }

    @Override
    public ModelSupport.ModelType getModelType() {
        return ModelSupport.getModelType("GEMMA4");
    }

    @Override
    protected TransformerBlock[] loadTransformerBlockWeights() {
        DType qType = modelQType.orElse(this.modelDType);
        if (qType != this.modelDType) {
            logger.info("Quantizing model with {} - Please hold...", qType);
        }

        TransformerBlock[] transformerBlocks = new TransformerBlock[c.dctx().numberOfLayers];

        IntStream.range(c.dctx().layerStart, c.dctx().layerEnd).parallel().forEach(i -> {
            String base = "model.language_model.layers." + i + ".";
            String prefix = base + "self_attn.";

            // Load attention weights
            // Gemma 4 uses attention_k_eq_v: V shares weights with K (no separate v_proj)
            // Also has q_norm and k_norm for QK normalization before RoPE
            var kProjWeight = weights.load(prefix + "k_proj.weight", c.dctx(), true, false).quantize(qType);

            // Load QK normalization weights
            AbstractTensor qNormWeight = weights.load(prefix + "q_norm.weight").quantize(qType);
            AbstractTensor kNormWeight = weights.load(prefix + "k_norm.weight").quantize(qType);

            Gemma4Attention attention = new Gemma4Attention(
                this,
                i,
                weights.load(prefix + "q_proj.weight", c.dctx(), true, false).quantize(qType),
                kProjWeight,
                kProjWeight, // V = K (attention_k_eq_v=true)
                weights.load(prefix + "o_proj.weight", c.dctx(), false, true).quantize(qType),
                qNormWeight,
                kNormWeight,
                c.layerNormEps
            );

            // Load MLP weights (SwiGLU: gate_proj, down_proj, up_proj)
            prefix = base + "mlp.";
            MLPBlock mlp = new MLPBlock(
                this,
                c.activationFunction,
                weights.load(prefix + "gate_proj.weight", c.dctx(), true, false).quantize(qType),
                weights.load(prefix + "down_proj.weight", c.dctx(), false, true).quantize(qType),
                weights.load(prefix + "up_proj.weight", c.dctx(), true, false).quantize(qType)
            );

            // Gemma 4 has pre/post norms for both attention and FFN (same as Gemma 2)
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

            // Log layer type for debugging
            if (((Gemma4Config) c).isGlobalLayer(i)) {
                logger.debug("Layer {} = GLOBAL (full attention)", i);
            } else {
                logger.debug("Layer {} = LOCAL (sliding window: {})", i, ((Gemma4Config) c).getSlidingWindowSize());
            }
        });

        return transformerBlocks;
    }

    @Override
    protected EmbedInput loadInputWeights() {
        if (wte == null) wte = weights.load("model.language_model.embed_tokens.weight").quantize(workingDType);

        return (inputToken, position) -> {
            AbstractTensor embedding = makeDenseTensor(c.embeddingLength);
            AbstractTensor at = wte.slice(true, inputToken);
            if (wte.dType() != embedding.dType()) {
                at = TensorOperationsProvider.get().quantize(at, embedding.dType(), 0, c.embeddingLength);
            }

            embedding.copyFrom(at, 0, 0, c.embeddingLength);

            // Gemma embedding scaling: multiply by sqrt(hidden_size)
            TensorOperationsProvider.get().scale(embeddingScalingFactor, embedding, 0, c.embeddingLength);

            return embedding;
        };
    }

    @Override
    protected SampleOutput loadOutputWeights() {
        DType qType = modelQType.orElse(this.modelDType);

        if (wte == null) wte = weights.load("model.language_model.embed_tokens.weight").quantize(workingDType);

        final LayerNorm layerNorm = new RMSNorm(this, weights.load("model.language_model.norm.weight").quantize(qType), 1.0f);

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
