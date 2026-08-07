package com.github.tjake.jlama.tensor.operations.opencl;

import com.github.tjake.jlama.model.*;
import com.github.tjake.jlama.safetensors.*;
import com.github.tjake.jlama.safetensors.tokenizer.*;
import com.github.tjake.jlama.safetensors.prompt.*;
import com.github.tjake.jlama.tensor.*;

import java.io.File;
import java.util.*;

import static com.github.tjake.jlama.model.ModelSupport.loadModel;

/**
 * GPU model runner. Uses Jlama's standard model loading, then extracts
 * weights and runs inference through GpuInferenceEngine.
 */
public class GpuModelRunner {

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.out.println("Usage: GpuModelRunner <model_dir> <prompt> [max_tokens]");
            System.exit(1);
        }
        File modelDir = new File(args[0]);
        String prompt = args[1];
        int maxTokens = args.length > 2 ? Integer.parseInt(args[2]) : 100;
        run(modelDir, prompt, maxTokens);
    }

    public static void run(File modelDir, String prompt, int maxTokens) throws Exception {
        System.out.println("[GPU] Loading model from: " + modelDir);

        // Use Jlama's standard model loading (handles config, tokenizer, weights)
        AbstractModel model = loadModel(
            modelDir, null,
            DType.F32, DType.I8,
            Optional.empty(), Optional.empty()
        );

        Config config = model.getConfig();
        Tokenizer tokenizer = model.getTokenizer();
        WeightLoader weights = model.getWeights();

        System.out.println("[GPU] Model loaded: " + config.embeddingLength + "d, "
            + config.numberOfLayers + " layers, vocab=" + config.vocabularySize);

        // Create and initialize GPU engine
        GpuInferenceEngine engine = new GpuInferenceEngine(config);
        engine.initialize();

        // Upload weights to GPU
        System.out.println("[GPU] Uploading weights to GPU...");
        long uploadStart = System.currentTimeMillis();
        uploadWeights(engine, weights, config);
        long uploadTime = System.currentTimeMillis() - uploadStart;
        System.out.println("[GPU] Weights uploaded in " + uploadTime + "ms");

        // Tokenize prompt
        long[] promptTokens = tokenizer.encode(prompt);
        System.out.println("[GPU] Prompt tokens: " + promptTokens.length);
        System.out.println("[GPU] Generating...\n");

        // Process prompt (prefill)
        long genStart = System.currentTimeMillis();
        int lastToken = 0;
        for (int i = 0; i < promptTokens.length; i++) {
            lastToken = engine.forward((int) promptTokens[i], i);
        }

        // Generate new tokens
        int generated = 0;
        int nextToken = lastToken;
        for (int i = 0; i < maxTokens; i++) {
            String text = tokenizer.decode(new long[]{nextToken});
            System.out.print(text);
            System.out.flush();
            generated++;

            if (nextToken == 128001 || nextToken == 128008 || nextToken == 128009) break;

            int pos = promptTokens.length + i;
            nextToken = engine.forward(nextToken, pos);
        }

        long genTime = System.currentTimeMillis() - genStart;
        System.out.println("\n\n[GPU] " + generated + " tokens in " + genTime + "ms ("
            + Math.round(generated * 1000.0 / genTime) + " tok/s)");
    }

    private static void uploadWeights(GpuInferenceEngine engine, WeightLoader weights, Config config) {
        // Embedding
        AbstractTensor embedTensor = weights.load("model.embed_tokens.weight");
        System.out.println("[GPU] embed_tokens: type=" + embedTensor.dType()
            + " shape=" + embedTensor.shape() + " dims=" + embedTensor.dims());
        engine.uploadEmbedding(embedTensor);

        AbstractTensor normTensor = weights.load("model.norm.weight");
        System.out.println("[GPU] norm: type=" + normTensor.dType()
            + " shape=" + normTensor.shape() + " dims=" + normTensor.dims());
        engine.uploadFinalNorm(normTensor);

        for (int i = 0; i < config.numberOfLayers; i++) {
            String p = "model.layers." + i + ".";

            engine.uploadF32Weight(weights.load(p + "input_layernorm.weight"),
                engine.getInputNormWeights(), i);
            engine.uploadF32Weight(weights.load(p + "post_attention_layernorm.weight"),
                engine.getPostNormWeights(), i);

            engine.uploadQ4Weight(weights.load(p + "self_attn.q_proj.weight"),
                engine.getQProjData(), engine.getQProjScales(), i);
            engine.uploadQ4Weight(weights.load(p + "self_attn.k_proj.weight"),
                engine.getKProjData(), engine.getKProjScales(), i);
            engine.uploadQ4Weight(weights.load(p + "self_attn.v_proj.weight"),
                engine.getVProjData(), engine.getVProjScales(), i);
            engine.uploadQ4Weight(weights.load(p + "self_attn.o_proj.weight"),
                engine.getOProjData(), engine.getOProjScales(), i);

            engine.uploadQ4Weight(weights.load(p + "mlp.gate_proj.weight"),
                engine.getGateProjData(), engine.getGateProjScales(), i);
            engine.uploadQ4Weight(weights.load(p + "mlp.up_proj.weight"),
                engine.getUpProjData(), engine.getUpProjScales(), i);
            engine.uploadQ4Weight(weights.load(p + "mlp.down_proj.weight"),
                engine.getDownProjData(), engine.getDownProjScales(), i);

            if (i % 4 == 0) System.out.println("[GPU]   Layer " + i + "/" + config.numberOfLayers);
        }
    }
}
