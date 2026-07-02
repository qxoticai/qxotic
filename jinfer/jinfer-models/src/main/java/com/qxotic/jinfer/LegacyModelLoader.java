// Legacy engine model loader: dispatches general.architecture to the ModelLegacy implementations
// (Llama3/Gemma4/Qwen35/GptOss/Nemotron/Llama). The architecture-agnostic GGUF loading plumbing
// (readGguf/loadTensors/loadQuantized/...) lives in ModelLoader (jinfer-kernels).
package com.qxotic.jinfer;

import com.qxotic.jinfer.jinja.JinjaRenderer;
import com.qxotic.format.gguf.GGUF;
import com.qxotic.format.gguf.TensorEntry;

import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.Map;

import static com.qxotic.jinfer.ModelLoader.f32OrNull;
import static com.qxotic.jinfer.ModelLoader.loadQuantized;
import static com.qxotic.jinfer.ModelLoader.loadTensors;
import static com.qxotic.jinfer.ModelLoader.readGguf;
import static com.qxotic.jinfer.ModelLoader.quantOrNull;
import static com.qxotic.jinfer.ModelLoader.ropeFreqFactors;
import static com.qxotic.jinfer.ModelLoader.toF32Tensor;

public final class LegacyModelLoader {

    private LegacyModelLoader() {
    }

    /** Loads a model, dispatching on {@code general.architecture}: {@code gemma4} -> {@link Gemma4},
     *  {@code qwen35}/{@code qwen35moe} -> {@link Qwen35}, {@code gpt-oss} -> {@link GptOss},
     *  {@code nemotron_h}/{@code nemotron_h_moe} -> {@link Nemotron},
     *  {@code llama}/{@code minicpm}/{@code mistral3}/{@code granite} -> {@link Llama3},
     *  {@code lfm*} (LFM2.5) -> {@link Llama}. */
    public static ModelLegacy loadModel(Path ggufPath, int contextLength) throws IOException {
        try (FileChannel fileChannel = FileChannel.open(ggufPath, StandardOpenOption.READ)) {
            GGUF gguf = readGguf(fileChannel, ggufPath.toString());
            String arch = gguf.getString("general.architecture");
            if (arch.equals("gemma4")) {
                try (var ignored = Timer.log("Load Gemma4 model")) {
                    return Gemma4.loadModel(fileChannel, gguf, contextLength, true);
                }
            }
            if (arch.equals("qwen35") || arch.equals("qwen35moe")) {
                try (var ignored = Timer.log("Load Qwen3.5 model")) {
                    return Qwen35.loadModel(fileChannel, gguf, contextLength, true);
                }
            }
            if (arch.equals("gpt-oss")) {
                try (var ignored = Timer.log("Load gpt-oss model")) {
                    return GptOss.loadModel(fileChannel, gguf, contextLength, true);
                }
            }
            if (arch.equals("nemotron_h_moe") || arch.equals("nemotron_h")) {
                try (var ignored = Timer.log("Load Nemotron model")) {
                    return Nemotron.loadModel(fileChannel, gguf, contextLength, true);
                }
            }
            // The standard Llama transformer (Llama 3.x). The Llama3 impl also covers same-graph
            // variants: MiniCPM (+ 3 scalars, default 1.0), Mistral-3/Ministral (YaRN RoPE +
            // attention temperature scaling), and Granite (dense; 4 scalars incl. a custom attention
            // scale). All distinguished by GGUF metadata, not extra classes. (Granite-hybrid/-moe use
            // their own architectures and are not handled here.)
            if (arch.equals("llama") || arch.equals("minicpm") || arch.equals("mistral3") || arch.equals("granite")) {
                try (var ignored = Timer.log("Load Llama model")) {
                    return Llama3.loadModel(fileChannel, gguf, contextLength, true);
                }
            }
            // The LFM loader reads lfm2.* metadata; routing any other architecture through it would
            // silently mis-load the weights and emit gibberish. Refuse unknown architectures so the
            // failure is a clear message instead.
            if (!arch.startsWith("lfm")) {
                throw new UnsupportedOperationException(
                        "Unsupported model architecture '" + arch + "' (supported: lfm2.5, gemma4, qwen35, gpt-oss, nemotron_h, nemotron_h_moe, llama, minicpm, mistral3, granite)");
            }
            try (var ignored = Timer.log("Load LFM25 model")) {
                return loadModel(fileChannel, gguf, contextLength, true);
            }
        }
    }

    static Llama loadModel(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeightsFlag) throws IOException {
        LFMTokenizer tokenizer = new LFMTokenizer(gguf, JinjaRenderer::template);

        String archPrefix = gguf.getString("general.architecture");

        int modelContextLength = gguf.getValue(int.class, archPrefix + ".context_length");
        if (contextLength < 0 || modelContextLength < contextLength) {
            contextLength = modelContextLength;
        }

        int embeddingLength = gguf.getValue(int.class, archPrefix + ".embedding_length");
        int numberOfHeads = gguf.getValue(int.class, archPrefix + ".attention.head_count");
        int numberOfLayers = gguf.getValue(int.class, archPrefix + ".block_count");

        int headSizeFull = embeddingLength / numberOfHeads;
        int headSizeSWA = headSizeFull;
        int slidingWindow = 1;
        float logitSoftcapping = gguf.getValueOrDefault(float.class, archPrefix + ".final_logit_softcapping", 0f);
        float rmsNormEps = gguf.getValueOrDefault(float.class, archPrefix + ".attention.layer_norm_rms_epsilon", 1e-6f);
        float ropeTheta = gguf.getValueOrDefault(float.class, archPrefix + ".rope.freq_base", 1000000f);
        float ropeThetaSWA = gguf.getValueOrDefault(float.class, archPrefix + ".rope.freq_base_swa", 10000f);

        // MoE parameters
        int expertCount = gguf.getValueOrDefault(int.class, archPrefix + ".expert_count", 0);
        int expertUsedCount = gguf.getValueOrDefault(int.class, archPrefix + ".expert_used_count", 0);
        int expertFeedForwardLength = gguf.getValueOrDefault(int.class, archPrefix + ".expert_feed_forward_length", 0);
        int leadingDenseBlockCount = gguf.getValueOrDefault(int.class, archPrefix + ".leading_dense_block_count", numberOfLayers);
        int expertGatingFunc = gguf.getValueOrDefault(int.class, archPrefix + ".expert_gating_func", 1); // 1=softmax, 2=sigmoid

        // Per-layer feed forward lengths (scalar for uniform, array for variable)
        int[] feedForwardLength;
        Object ffnRaw = gguf.getValue(Object.class, archPrefix + ".feed_forward_length");
        if (ffnRaw instanceof int[] arr) {
            feedForwardLength = arr;
        } else {
            feedForwardLength = new int[numberOfLayers];
            Arrays.fill(feedForwardLength, (int) ffnRaw);
        }

        boolean[] isSWA = new boolean[numberOfLayers];

        int[] numberOfKeyValueHeadsPerLayer;
        Object kvHeads = gguf.getValue(Object.class, archPrefix + ".attention.head_count_kv");
        if (kvHeads instanceof int[] arr && arr.length == numberOfLayers) {
            numberOfKeyValueHeadsPerLayer = arr;
        } else {
            numberOfKeyValueHeadsPerLayer = new int[numberOfLayers];
            for (int i = 0; i < numberOfLayers; i++) {
                TensorEntry kWeight = gguf.getTensor("blk." + i + ".attn_k.weight");
                numberOfKeyValueHeadsPerLayer[i] = kWeight != null ? Math.toIntExact(kWeight.shape()[1]) / headSizeFull : 0;
            }
        }
        int shortConvLCache = gguf.getValueOrDefault(int.class, archPrefix + ".shortconv.l_cache", 0);

        Llama.Configuration config = new Llama.Configuration(
                embeddingLength,
                feedForwardLength,
                numberOfLayers,
                numberOfHeads,
                numberOfKeyValueHeadsPerLayer,
                tokenizer.vocabularySize(),
                contextLength,
                rmsNormEps,
                ropeTheta,
                ropeThetaSWA,
                headSizeFull,
                headSizeSWA,
                slidingWindow,
                logitSoftcapping,
                isSWA,
                numberOfLayers,
                expertCount,
                expertUsedCount,
                expertFeedForwardLength,
                shortConvLCache,
                leadingDenseBlockCount,
                expertGatingFunc
        );

        if (!loadWeightsFlag) {
            return new Llama(config, tokenizer, null);
        }

        Map<String, GGMLTensorEntry> tensorEntries = loadTensors(fileChannel, gguf);
        Llama.Weights qw = loadWeights(tensorEntries, config);
        return new Llama(config, tokenizer, qw);
    }

    public static Llama.Weights loadWeights(Map<String, GGMLTensorEntry> tensorEntries, Llama.Configuration config) {
        Pair<float[], float[]> ropeFreqsSWA = RoPE.precomputeFreqsCis(config.contextLength, config.headSizeSWA, config.ropeThetaSWA);
        float[] modelRopeFreqs = ropeFreqFactors(tensorEntries);
        Pair<float[], float[]> ropeFreqsFull = modelRopeFreqs != null
                ? RoPE.precomputeFreqsCisFromFreqs(config.contextLength, config.headSizeFull, config.ropeTheta, modelRopeFreqs)
                : RoPE.precomputeFreqsCis(config.contextLength, config.headSizeFull, config.ropeTheta);
        return loadWeightsWithRoPE(tensorEntries, config, ropeFreqsSWA, ropeFreqsFull);
    }

    public static Llama.Weights loadWeightsWithRoPE(Map<String, GGMLTensorEntry> tensorEntries, Llama.Configuration config,
                                                     Pair<float[], float[]> ropeFreqsSWA, Pair<float[], float[]> ropeFreqsFull) {
        FloatTensor tokenEmbeddingTable = loadQuantized(tensorEntries.get("token_embd.weight"));
        Llama.LayerWeights[] layers = new Llama.LayerWeights[config.numberOfLayers];
        for (int i = 0; i < config.numberOfLayers; i++) {
            layers[i] = loadLayer(tensorEntries, "blk." + i + ".", config.embeddingLength);
        }
        return new Llama.Weights(
                tokenEmbeddingTable,
                layers,
                toF32Tensor(tensorEntries.getOrDefault("output_norm.weight", tensorEntries.get("token_embd_norm.weight"))),
                F32FloatTensor.of(ropeFreqsFull.first()), F32FloatTensor.of(ropeFreqsFull.second()),
                F32FloatTensor.of(ropeFreqsSWA.first()), F32FloatTensor.of(ropeFreqsSWA.second()),
                tensorEntries.containsKey("output.weight")
                        ? loadQuantized(tensorEntries.get("output.weight"))
                        : tokenEmbeddingTable);
    }

    private static Llama.LayerWeights loadLayer(Map<String, GGMLTensorEntry> entries, String prefix, int dim) {
        Llama.AttentionWeights attention = !entries.containsKey(prefix + "attn_q.weight") ? null
                : new Llama.AttentionWeights(
                        loadQuantized(entries.get(prefix + "attn_q.weight")),
                        loadQuantized(entries.get(prefix + "attn_k.weight")),
                        quantOrNull(entries, prefix + "attn_v.weight"),
                        loadQuantized(entries.get(prefix + "attn_output.weight")),
                        f32OrNull(entries, prefix + "attn_q_norm.weight"),
                        f32OrNull(entries, prefix + "attn_k_norm.weight"));
        Llama.ShortConvWeights shortConv = !entries.containsKey(prefix + "shortconv.conv.weight") ? null
                : Llama.ShortConvWeights.of(
                        toF32Tensor(entries.get(prefix + "shortconv.conv.weight")),
                        loadQuantized(entries.get(prefix + "shortconv.in_proj.weight")),
                        loadQuantized(entries.get(prefix + "shortconv.out_proj.weight")),
                        dim);
        Llama.DenseFfnWeights dense = !entries.containsKey(prefix + "ffn_gate.weight") ? null
                : new Llama.DenseFfnWeights(
                        loadQuantized(entries.get(prefix + "ffn_gate.weight")),
                        loadQuantized(entries.get(prefix + "ffn_down.weight")),
                        loadQuantized(entries.get(prefix + "ffn_up.weight")));
        Llama.MoeFfnWeights moe = !entries.containsKey(prefix + "ffn_gate_inp.weight") ? null
                : new Llama.MoeFfnWeights(
                        loadQuantized(entries.get(prefix + "ffn_gate_inp.weight")),
                        loadQuantized(entries.get(prefix + "ffn_gate_exps.weight")),
                        loadQuantized(entries.get(prefix + "ffn_up_exps.weight")),
                        loadQuantized(entries.get(prefix + "ffn_down_exps.weight")),
                        f32OrNull(entries, prefix + "exp_probs_b.bias"));
        GGMLTensorEntry scaleEntry = entries.get(prefix + "layer_output_scale.weight");
        return new Llama.LayerWeights(
                toF32Tensor(entries.get(prefix + "attn_norm.weight")),
                f32OrNull(entries, prefix + "post_attention_norm.weight"),
                toF32Tensor(entries.get(prefix + "ffn_norm.weight")),
                f32OrNull(entries, prefix + "post_ffw_norm.weight"),
                scaleEntry != null ? toF32Tensor(scaleEntry).getFloat(0) : 1.0f,
                attention, shortConv, dense, moe);
    }
}
