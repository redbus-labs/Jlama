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

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.github.tjake.jlama.math.ActivationFunction;
import com.github.tjake.jlama.safetensors.Config;
import java.util.List;
import java.util.Map;

/**
 * Configuration for the Gemma 4 Unified model.
 *
 * Gemma 4's config.json has a NESTED structure:
 * - Top level: model_type, text_config, vision_config, audio_config
 * - text_config: contains all the transformer parameters (hidden_size, num_layers, etc.)
 * - layer_types: array specifying "sliding_attention" or "full_attention" per layer
 *
 * This config reads from text_config and exposes the layer type information.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class Gemma4Config extends Config {

    private final List<String> layerTypes;
    private final int slidingWindowSize;

    @JsonCreator
    public Gemma4Config(
        @JsonProperty("text_config") TextConfig textConfig,
        @JsonProperty("eos_token_id") Object eosTokens
    ) {
        super(
            textConfig.contextLength,
            textConfig.embeddingLength,
            textConfig.hiddenLength,
            textConfig.numberOfHeads,
            textConfig.numberOfKeyValueHeads,
            textConfig.numberOfLayers,
            textConfig.layerNormEps,
            textConfig.vocabularySize,
            textConfig.bosToken,
            eosTokens instanceof List ? (List<Integer>) eosTokens : List.of(1, 106),
            ActivationFunction.Type.GELU,
            textConfig.ropeTheta != null ? textConfig.ropeTheta : 10000.0,
            1.0,
            textConfig.headDim,
            textConfig.finalLogitSoftCapping,
            null // no attn softcapping at config level
        );
        this.layerTypes = textConfig.layerTypes;
        this.slidingWindowSize = textConfig.slidingWindow != null ? textConfig.slidingWindow : 1024;
    }

    /**
     * Returns the sliding window size for local attention layers.
     */
    public int getSlidingWindowSize() {
        return slidingWindowSize;
    }

    /**
     * Determines if a given layer uses global (full) attention.
     * Uses the layer_types array from config: "full_attention" = global, "sliding_attention" = local.
     */
    public boolean isGlobalLayer(int layerIndex) {
        if (layerTypes == null || layerIndex >= layerTypes.size()) {
            // Fallback: last layer is global, odd layers are global
            return layerIndex == numberOfLayers - 1 || layerIndex % 2 == 1;
        }
        return "full_attention".equals(layerTypes.get(layerIndex));
    }

    /**
     * Nested text_config structure from Gemma 4's config.json
     */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class TextConfig {
        public final int contextLength;
        public final int embeddingLength;
        public final int hiddenLength;
        public final int numberOfHeads;
        public final int numberOfKeyValueHeads;
        public final int numberOfLayers;
        public final float layerNormEps;
        public final int vocabularySize;
        public final int bosToken;
        public final Integer headDim;
        public final Integer slidingWindow;
        public final Double ropeTheta;
        public final Float finalLogitSoftCapping;
        public final List<String> layerTypes;

        @JsonCreator
        public TextConfig(
            @JsonProperty("max_position_embeddings") int contextLength,
            @JsonProperty("hidden_size") int embeddingLength,
            @JsonProperty("intermediate_size") int hiddenLength,
            @JsonProperty("num_attention_heads") int numberOfHeads,
            @JsonProperty("num_key_value_heads") int numberOfKeyValueHeads,
            @JsonProperty("num_hidden_layers") int numberOfLayers,
            @JsonProperty("rms_norm_eps") float layerNormEps,
            @JsonProperty("vocab_size") int vocabularySize,
            @JsonProperty("bos_token_id") int bosToken,
            @JsonProperty("head_dim") Integer headDim,
            @JsonProperty("sliding_window") Integer slidingWindow,
            @JsonProperty("rope_theta") Double ropeTheta,
            @JsonProperty("final_logit_softcapping") Float finalLogitSoftCapping,
            @JsonProperty("layer_types") List<String> layerTypes
        ) {
            this.contextLength = contextLength;
            this.embeddingLength = embeddingLength;
            this.hiddenLength = hiddenLength;
            this.numberOfHeads = numberOfHeads;
            this.numberOfKeyValueHeads = numberOfKeyValueHeads;
            this.numberOfLayers = numberOfLayers;
            this.layerNormEps = layerNormEps;
            this.vocabularySize = vocabularySize;
            this.bosToken = bosToken;
            this.headDim = headDim;
            this.slidingWindow = slidingWindow;
            this.ropeTheta = ropeTheta;
            this.finalLogitSoftCapping = finalLogitSoftCapping;
            this.layerTypes = layerTypes;
        }
    }
}
