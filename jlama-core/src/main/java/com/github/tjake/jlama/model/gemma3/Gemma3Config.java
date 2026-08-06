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

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.github.tjake.jlama.math.ActivationFunction;
import com.github.tjake.jlama.safetensors.Config;
import java.util.List;
import java.util.Map;

/**
 * Configuration for Google's Gemma 3 model family.
 *
 * Gemma 3 uses a nested config structure (text_config + vision_config) and
 * has hybrid attention: alternating 5 local sliding window layers per 1 global layer.
 * Uses query_pre_attn_scalar for attention score scaling.
 *
 * Key differences from Gemma 2:
 * - Nested text_config in config.json
 * - layer_types array: "sliding_attention" / "full_attention"
 * - sliding_window = 4096 (vs none in Gemma 2)
 * - 128K context window
 * - query_pre_attn_scalar for attention scaling
 * - Has separate v_proj (unlike Gemma 4 which shares K=V)
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class Gemma3Config extends Config {

    private final List<String> layerTypes;
    private final int slidingWindowSize;
    private final float queryPreAttnScalar;

    @JsonCreator
    public Gemma3Config(
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
            eosTokens instanceof List ? (List<Integer>) eosTokens : List.of(1),
            ActivationFunction.Type.GELU,
            textConfig.ropeTheta != null ? textConfig.ropeTheta : 10000.0,
            1.0,
            textConfig.headDim,
            textConfig.finalLogitSoftCapping,
            textConfig.attnLogitSoftCapping
        );
        this.layerTypes = textConfig.layerTypes;
        this.slidingWindowSize = textConfig.slidingWindow != null ? textConfig.slidingWindow : 4096;
        this.queryPreAttnScalar = textConfig.queryPreAttnScalar != null ? textConfig.queryPreAttnScalar : 256.0f;
    }

    public int getSlidingWindowSize() {
        return slidingWindowSize;
    }

    public float getQueryPreAttnScalar() {
        return queryPreAttnScalar;
    }

    /**
     * Determines if a layer uses global (full) attention.
     * Gemma 3 pattern: 5 sliding window layers then 1 global layer, repeating.
     */
    public boolean isGlobalLayer(int layerIndex) {
        if (layerTypes == null || layerIndex >= layerTypes.size()) {
            // Fallback: every 6th layer is global (5 local + 1 global pattern)
            return (layerIndex + 1) % 6 == 0;
        }
        return "full_attention".equals(layerTypes.get(layerIndex));
    }

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
        public final Float attnLogitSoftCapping;
        public final Float queryPreAttnScalar;
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
            @JsonProperty("attn_logit_softcapping") Float attnLogitSoftCapping,
            @JsonProperty("query_pre_attn_scalar") Float queryPreAttnScalar,
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
            this.attnLogitSoftCapping = attnLogitSoftCapping;
            this.queryPreAttnScalar = queryPreAttnScalar;
            this.layerTypes = layerTypes;
        }
    }
}
