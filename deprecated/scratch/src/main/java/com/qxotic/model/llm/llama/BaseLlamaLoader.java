package com.qxotic.model.llm.llama;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.format.gguf.TensorEntry;
import com.qxotic.model.llm.AbstractGGUFLoader;
import com.qxotic.model.llm.ChatFormat;
import com.qxotic.model.llm.SpanLoader;
import com.qxotic.span.FloatMatrixView;
import com.qxotic.span.FloatSpan;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.impl.Tiktoken;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.function.IntFunction;
import java.util.regex.Pattern;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class BaseLlamaLoader
        extends AbstractGGUFLoader<Llama, Llama.Configuration, Llama.Weights, Llama.State> {

    public BaseLlamaLoader(GGUF gguf) {
        super(gguf);
    }

    @Override
    public Llama loadModel(Llama.Configuration configuration) {
        return new Llama(
                configuration, DefaultKernelOps.getKernelOps(), DefaultKernelOps.getSpanFactory());
    }

    @Override
    public ChatFormat createChatFormat(Tokenizer tokenizer) {
        return new Llama3ChatFormat(tokenizer);
    }

    @Override
    public Llama.Configuration loadConfiguration(int maxTokens, SpanLoader spanLoader) {
        String arch = gguf.getValue(String.class, "general.architecture");

        int embeddingLength = gguf.getValue(int.class, arch + ".embedding_length");
        int numberOfHeads = gguf.getValue(int.class, arch + ".attention.head_count");

        int keyHeadSize =
                gguf.getValueOrDefault(
                        int.class, arch + ".attention.key_length", embeddingLength / numberOfHeads);
        int valueHeadSize =
                gguf.getValueOrDefault(
                        int.class,
                        arch + ".attention.value_length",
                        embeddingLength / numberOfHeads);

        return new Llama.Configuration(
                        embeddingLength,
                        gguf.getValue(int.class, arch + ".feed_forward_length"),
                        gguf.getValue(int.class, arch + ".block_count"),
                        numberOfHeads,
                        keyHeadSize,
                        valueHeadSize,
                        gguf.containsKey(arch + ".attention.head_count_kv")
                                ? gguf.getValue(int.class, arch + ".attention.head_count_kv")
                                : numberOfHeads,
                        (gguf.getValue(String[].class, "tokenizer.ggml.tokens")).length,
                        gguf.getValue(int.class, arch + ".context_length"),
                        gguf.getValueOrDefault(
                                float.class, arch + ".attention.layer_norm_rms_epsilon", 1e-5f),
                        gguf.getValueOrDefault(float.class, arch + ".rope.freq_base", 10000f),
                        false // Llama 3 uses classic RoPE
                        )
                .with(b -> b.contextLength(maxTokens));
    }

    @Override
    public Llama.Weights loadWeights(Llama.Configuration config, SpanLoader spanLoader) {
        try (var timer = Timer.log("Load weights")) {

            float[][] ropeFreqs;
            if (gguf.containsTensor("rope_freqs.weight")) {
                FloatSpan ropeFreqsWeightSpan =
                        spanLoader.apply(gguf.getTensor("rope_freqs.weight"));
                assert ropeFreqsWeightSpan.size() == config.keyHeadSize / 2;
                float[] ropeFreqsValues = new float[config.keyHeadSize / 2];
                DefaultKernelOps.getKernelOps()
                        .copyTo(ropeFreqsWeightSpan, ArraySpan.wrap(ropeFreqsValues));
                ropeFreqs =
                        RoPE.precomputeFreqsCis(
                                config.contextLength,
                                config.keyHeadSize,
                                config.ropeTheta,
                                ropeFreqsValues);
            } else {
                ropeFreqs =
                        RoPE.precomputeFreqsCis(
                                config.contextLength, config.keyHeadSize, config.ropeTheta);
            }

            assert ropeFreqs.length == 2;
            float[] ropeFreqsReal = ropeFreqs[0];
            float[] ropeFreqsImag = ropeFreqs[1];

            FloatMatrixView tokenEmbeddings =
                    tensorAsMatrix(gguf.getTensor("token_embd.weight"), spanLoader, true);

            // If "output.weight" is not present then the embedding weights are tied/shared with the
            // decoder.
            // This is commonly referred as "tie word embeddings".
            FloatMatrixView classifierWeights =
                    gguf.containsTensor("output.weight")
                            ? tensorAsMatrix(gguf.getTensor("output.weight"), spanLoader, true)
                            : tokenEmbeddings;

            return new Llama.Weights(
                    tokenEmbeddings,
                    loadSpanArray(
                            config.numberOfLayers,
                            i -> gguf.getTensor("blk." + i + ".attn_norm.weight"),
                            spanLoader),
                    loadMatrixArray(
                            false,
                            config.numberOfLayers,
                            i -> gguf.getTensor("blk." + i + ".attn_q.weight"),
                            spanLoader),
                    loadMatrixArray(
                            false,
                            config.numberOfLayers,
                            i -> gguf.getTensor("blk." + i + ".attn_k.weight"),
                            spanLoader),
                    loadMatrixArray(
                            false,
                            config.numberOfLayers,
                            i -> gguf.getTensor("blk." + i + ".attn_v.weight"),
                            spanLoader),

                    // qk norm (Qwen 3)
                    loadSpanArray(
                            true,
                            config.numberOfLayers,
                            i -> gguf.getTensor("blk." + i + ".attn_q_norm.weight"),
                            spanLoader),
                    loadSpanArray(
                            true,
                            config.numberOfLayers,
                            i -> gguf.getTensor("blk." + i + ".attn_k_norm.weight"),
                            spanLoader),

                    // qkv bias can be null.
                    loadSpanArray(
                            true,
                            config.numberOfLayers,
                            i -> gguf.getTensor("blk." + i + ".attn_q.bias"),
                            spanLoader),
                    loadSpanArray(
                            true,
                            config.numberOfLayers,
                            i -> gguf.getTensor("blk." + i + ".attn_k.bias"),
                            spanLoader),
                    loadSpanArray(
                            true,
                            config.numberOfLayers,
                            i -> gguf.getTensor("blk." + i + ".attn_v.bias"),
                            spanLoader),
                    loadMatrixArray(
                            false,
                            config.numberOfLayers,
                            i -> gguf.getTensor("blk." + i + ".attn_output.weight"),
                            spanLoader),
                    loadSpanArray(
                            config.numberOfLayers,
                            i -> gguf.getTensor("blk." + i + ".ffn_norm.weight"),
                            spanLoader),
                    loadMatrixArray(
                            true,
                            config.numberOfLayers,
                            i -> gguf.getTensor("blk." + i + ".ffn_gate.weight"),
                            spanLoader),
                    loadMatrixArray(
                            false,
                            config.numberOfLayers,
                            i -> gguf.getTensor("blk." + i + ".ffn_down.weight"),
                            spanLoader),
                    loadMatrixArray(
                            false,
                            config.numberOfLayers,
                            i -> gguf.getTensor("blk." + i + ".ffn_up.weight"),
                            spanLoader),
                    spanLoader.apply(gguf.getTensor("output_norm.weight")),
                    ArraySpan.wrap(ropeFreqsReal),
                    ArraySpan.wrap(ropeFreqsImag),
                    classifierWeights);
        }
    }

    public static FloatMatrixView tensorAsMatrix(
            TensorEntry tensorEntry, SpanLoader spanLoader, boolean precomputeRowSpans) {
        long rows = tensorEntry.shape()[1];
        long cols = tensorEntry.shape()[0];
        if (tensorEntry.shape().length != 2) {
            throw new IllegalArgumentException();
        }
        FloatSpan span = spanLoader.apply(tensorEntry);
        if (rows * cols != span.size()) {
            throw new IllegalArgumentException();
        }
        return FloatMatrixView.asMatrix(span, 0, rows, cols, cols, precomputeRowSpans);
    }

    public static FloatMatrixView tensorAsMatrix(TensorEntry tensorEntry, SpanLoader spanLoader) {
        return tensorAsMatrix(tensorEntry, spanLoader, false);
    }

    public static FloatSpan[] loadSpanArray(
            int size,
            IntFunction<TensorEntry> getTensorInfo,
            Function<Object, FloatSpan> spanLoader) {
        return loadSpanArray(false, size, getTensorInfo, spanLoader);
    }

    public static FloatMatrixView[] loadMatrixArray(
            boolean isOptional,
            int size,
            IntFunction<TensorEntry> getTensorInfo,
            SpanLoader spanLoader) {
        // return IntStream.range(0,
        // size).mapToObj(getTensorInfo).map(spanLoader).toArray(FloatSpan[]::new);
        FloatMatrixView[] array =
                null; // Lazily allocated if there's at it contains at least one entry.
        for (int i = 0; i < size; i++) {
            TensorEntry tensorEntry = getTensorInfo.apply(i);
            if (tensorEntry != null) {
                if (array == null) {
                    array = new FloatMatrixView[size];
                }
                array[i] = tensorAsMatrix(tensorEntry, spanLoader);
            }
        }
        if (array == null && !isOptional) {
            throw new IllegalArgumentException("non-optional tensor wasn't found");
        }
        return array;
    }

    public static FloatSpan[] loadSpanArray(
            boolean isOptional,
            int size,
            IntFunction<TensorEntry> getTensorInfo,
            Function<Object, FloatSpan> spanLoader) {
        // return IntStream.range(0,
        // size).mapToObj(getTensorInfo).map(spanLoader).toArray(FloatSpan[]::new);
        FloatSpan[] array = null; // Lazily allocated if there's at it contains at least one entry.
        for (int i = 0; i < size; i++) {
            TensorEntry tensorEntry = getTensorInfo.apply(i);
            if (tensorEntry != null) {
                if (array == null) {
                    array = new FloatSpan[size];
                }
                array[i] = spanLoader.apply(tensorEntry);
            }
        }
        if (array == null && !isOptional) {
            throw new IllegalArgumentException("non-optional tensor wasn't found");
        }
        return array;
    }

    public static Tokenizer loadTokenizerFromTiktoken(Path tiktokenModel) throws IOException {
        try (var reader = Files.newBufferedReader(tiktokenModel)) {
            Map<String, Integer> mergeableRanks = Tiktoken.loadMergeableRanks(reader);
            return loadTokenizerFromTiktoken(mergeableRanks);
        }
    }

    public static Tokenizer loadTokenizerFromTiktoken(Map<String, Integer> mergeableRanks) {
        Map<String, Integer> specialTokens = llama3SpecialTokens(mergeableRanks);
        return Tiktoken.createFromTiktoken(
                "llama3",
                mergeableRanks,
                Pattern.compile(LlamaBPETextSplitterFactory.LLAMA3_PATTERN),
                specialTokens);
    }

    private static Map<String, Integer> llama3SpecialTokens(Map<String, Integer> mergeableRanks) {
        int numBaseTokens = mergeableRanks.size();
        int numReservedSpecialTokens = 256;

        List<String> specialTokensList =
                Stream.concat(
                                Stream.of(
                                        "<|begin_of_text|>",
                                        "<|end_of_text|>",
                                        "<|reserved_special_token_0|>",
                                        "<|reserved_special_token_1|>",
                                        "<|reserved_special_token_2|>",
                                        "<|reserved_special_token_3|>",
                                        "<|start_header_id|>",
                                        "<|end_header_id|>",
                                        "<|reserved_specialtorch_token_4|>",
                                        "<|eot_id|>" // end of turn
                                        ),
                                IntStream.range(5, numReservedSpecialTokens - 5)
                                        .mapToObj(i -> "<|reserved_special_token_" + i + "|>"))
                        .toList();

        Map<String, Integer> specialTokens = new HashMap<>(specialTokensList.size());
        for (int i = 0; i < specialTokensList.size(); ++i) {
            String token = specialTokensList.get(i);
            specialTokens.put(token, numBaseTokens + i);
        }

        return specialTokens;
    }
}
