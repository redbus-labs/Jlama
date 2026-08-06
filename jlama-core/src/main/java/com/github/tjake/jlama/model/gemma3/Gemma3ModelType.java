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

import com.github.tjake.jlama.model.AbstractModel;
import com.github.tjake.jlama.model.ModelSupport;
import com.github.tjake.jlama.model.gemma.GemmaTokenizer;
import com.github.tjake.jlama.safetensors.Config;
import com.github.tjake.jlama.safetensors.tokenizer.Tokenizer;

/**
 * Model type registration for Gemma 3.
 * Reuses GemmaTokenizer since Gemma 3 uses the same <start_of_turn> chat template.
 */
public class Gemma3ModelType implements ModelSupport.ModelType {
    @Override
    public Class<? extends AbstractModel> getModelClass() {
        return Gemma3Model.class;
    }

    @Override
    public Class<? extends Config> getConfigClass() {
        return Gemma3Config.class;
    }

    @Override
    public Class<? extends Tokenizer> getTokenizerClass() {
        return GemmaTokenizer.class;
    }
}
