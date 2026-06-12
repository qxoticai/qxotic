// Model loading: GGUF metadata/tensors via com.qxotic:gguf, the toknroll tokenizer adapter,
// and the conversion into the engine's Configuration / per-layer Weights.
package com.llama4j;

import com.qxotic.format.gguf.GGMLType;
import com.qxotic.format.gguf.GGUF;
import com.qxotic.format.gguf.TensorEntry;
import com.qxotic.toknroll.Normalizer;
import com.qxotic.toknroll.Specials;
import com.qxotic.toknroll.StandardTokenType;
import com.qxotic.toknroll.Splitter;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.gguf.GGUFTokenizerLoader;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorShape;
import jdk.incubator.vector.VectorSpecies;

import com.sun.net.httpserver.Headers;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

import java.lang.reflect.Field;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.PrintStream;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.net.InetSocketAddress;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.CharBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CharsetDecoder;
import java.nio.charset.CodingErrorAction;
import java.nio.channels.FileChannel;
import java.nio.channels.ReadableByteChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.OptionalInt;
import java.util.Set;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;
import java.util.function.IntConsumer;
import java.util.function.IntFunction;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

interface Timer extends AutoCloseable {
    @Override
    void close(); // no Exception

    static Timer log(String label) {
        return log(label, TimeUnit.MILLISECONDS);
    }

    static Timer log(String label, TimeUnit timeUnit) {
        return new Timer() {
            final long startNanos = System.nanoTime();

            @Override
            public void close() {
                long elapsedNanos = System.nanoTime() - startNanos;
                System.err.println(label + ": "
                        + timeUnit.convert(elapsedNanos, TimeUnit.NANOSECONDS) + " "
                        + timeUnit.toChronoUnit().name().toLowerCase());
            }
        };
    }
}

final class ModelLoader {

    /** Parses the GGUF metadata (com.qxotic:gguf) from the channel, leaving its position past the header. */
    static GGUF readGguf(FileChannel fileChannel, String modelLabel) throws IOException {
        try (var ignored = Timer.log("Parse " + modelLabel)) {
            fileChannel.position(0L);
            return GGUF.read(java.nio.channels.Channels.newChannel(
                    new java.io.BufferedInputStream(java.nio.channels.Channels.newInputStream(fileChannel), 1 << 20)));
        }
    }

    /** Memory-maps the tensor data (whole-file mapping outlives the process: Arena.global). */
    static Map<String, GGMLTensorEntry> loadTensors(FileChannel fileChannel, GGUF gguf) throws IOException {
        return loadTensors(fileChannel, gguf.getTensorDataOffset(), gguf.getTensors());
    }

    static Map<String, GGMLTensorEntry> loadTensors(FileChannel fileChannel, long tensorDataOffset,
                                                    java.util.Collection<TensorEntry> tensors) throws IOException {
        MemorySegment tensorData = fileChannel.map(FileChannel.MapMode.READ_ONLY, tensorDataOffset,
                fileChannel.size() - tensorDataOffset, Arena.global());
        Map<String, GGMLTensorEntry> tensorEntries = HashMap.newHashMap(tensors.size());
        for (TensorEntry tensor : tensors) {
            int[] shape = Arrays.stream(tensor.shape()).mapToInt(Math::toIntExact).toArray();
            long sizeInBytes = tensor.ggmlType().byteSizeFor(FloatTensor.numberOfElementsLong(shape));
            MemorySegment memorySegment = tensorData.asSlice(tensor.offset(), sizeInBytes);
            tensorEntries.put(tensor.name(), new GGMLTensorEntry(tensor.name(), tensor.ggmlType(), shape, memorySegment));
        }
        return tensorEntries;
    }

    public static Llama loadModel(Path ggufPath, int contextLength) throws IOException {
        try (var ignored = Timer.log("Load LFM25 model")) {
            try (FileChannel fileChannel = FileChannel.open(ggufPath, StandardOpenOption.READ)) {
                GGUF gguf = readGguf(fileChannel, ggufPath.toString());
                return loadModel(fileChannel, gguf, contextLength, true);
            }
        }
    }

    static Llama loadModel(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeightsFlag) throws IOException {
        LFMTokenizer tokenizer = new LFMTokenizer(gguf);

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
        int nLayerKvFromStart = numberOfLayers;
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
                nLayerKvFromStart,
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
        GGMLTensorEntry ropeFreqEntry = tensorEntries.get("rope_freqs.weight");
        Pair<float[], float[]> ropeFreqsFull;
        if (ropeFreqEntry != null) {
            float[] modelRopeFreqs = ropeFreqEntry.memorySegment().toArray(ValueLayout.JAVA_FLOAT);
            ropeFreqsFull = RoPE.precomputeFreqsCisFromFreqs(config.contextLength, config.headSizeFull, config.ropeTheta, modelRopeFreqs);
        } else {
            ropeFreqsFull = RoPE.precomputeFreqsCis(config.contextLength, config.headSizeFull, config.ropeTheta);
        }
        return loadWeightsWithRoPE(tensorEntries, config, ropeFreqsSWA, ropeFreqsFull);
    }

    public static Llama.Weights loadWeightsWithRoPE(Map<String, GGMLTensorEntry> tensorEntries, Llama.Configuration config,
                                                     Pair<float[], float[]> ropeFreqsSWA, Pair<float[], float[]> ropeFreqsFull) {
        FloatTensor tokenEmbeddingTable = loadQuantized(tensorEntries.get("token_embd.weight"));
        Llama.LayerWeights[] layers = new Llama.LayerWeights[config.numberOfLayers];
        for (int i = 0; i < config.numberOfLayers; i++) {
            layers[i] = loadLayer(tensorEntries, "blk." + i + ".");
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

    private static Llama.LayerWeights loadLayer(Map<String, GGMLTensorEntry> entries, String prefix) {
        Llama.AttentionWeights attention = !entries.containsKey(prefix + "attn_q.weight") ? null
                : new Llama.AttentionWeights(
                        loadQuantized(entries.get(prefix + "attn_q.weight")),
                        loadQuantized(entries.get(prefix + "attn_k.weight")),
                        quantOrNull(entries, prefix + "attn_v.weight"),
                        loadQuantized(entries.get(prefix + "attn_output.weight")),
                        f32OrNull(entries, prefix + "attn_q_norm.weight"),
                        f32OrNull(entries, prefix + "attn_k_norm.weight"));
        Llama.ShortConvWeights shortConv = !entries.containsKey(prefix + "shortconv.conv.weight") ? null
                : new Llama.ShortConvWeights(
                        toF32Tensor(entries.get(prefix + "shortconv.conv.weight")),
                        loadQuantized(entries.get(prefix + "shortconv.in_proj.weight")),
                        loadQuantized(entries.get(prefix + "shortconv.out_proj.weight")));
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

    private static FloatTensor quantOrNull(Map<String, GGMLTensorEntry> entries, String name) {
        GGMLTensorEntry entry = entries.get(name);
        return entry != null ? loadQuantized(entry) : null;
    }

    private static F32FloatTensor f32OrNull(Map<String, GGMLTensorEntry> entries, String name) {
        GGMLTensorEntry entry = entries.get(name);
        return entry != null ? toF32Tensor(entry) : null;
    }


    public static FloatTensor loadQuantized(GGMLTensorEntry entry) {
        GGMLType ggmlType = entry.ggmlType();
        long numElements = FloatTensor.numberOfElementsLong(entry.shape());
        return switch (ggmlType) {
            case Q8_0 -> new Q8_0FloatTensor(numElements, entry.memorySegment());
            case Q4_0 -> new Q4_0FloatTensor(numElements, entry.memorySegment());
            case Q4_1 -> new Q4_1FloatTensor(numElements, entry.memorySegment());
            case Q5_1 -> new Q5_1FloatTensor(numElements, entry.memorySegment());
            case Q4_K -> new Q4_KFloatTensor(numElements, entry.memorySegment());
            case Q5_K -> new Q5_KFloatTensor(numElements, entry.memorySegment());
            case Q6_K -> new Q6_KFloatTensor(numElements, entry.memorySegment());
            case F32 -> new F32FloatTensor(numElements, entry.memorySegment());
            case F16 -> new F16FloatTensor(numElements, entry.memorySegment());
            case BF16 -> new BF16FloatTensor(numElements, entry.memorySegment());
            case MXFP4 -> new MXFP4FloatTensor(numElements, entry.memorySegment());
            default -> throw new UnsupportedOperationException("Quantization format " + ggmlType);
        };
    }

    /** Zero-copy F32 view of a GGUF tensor (native mapped segment). */
    public static F32FloatTensor toF32Tensor(GGMLTensorEntry tensorEntry) {
        GGMLType ggmlType = tensorEntry.ggmlType();
        if (ggmlType != GGMLType.F32) {
            throw new UnsupportedOperationException("Conversion to " + ggmlType);
        }
        return new F32FloatTensor(FloatTensor.numberOfElementsLong(tensorEntry.shape()), tensorEntry.memorySegment());
    }
}

/**
 * LFM2 tokenizer = com.qxotic:toknroll loaded from the GGUF metadata, plus the "lfm2"
 * pre-tokenizer registration (llama.cpp's regex; [\p{L}\p{M}]+ because Java's \p{L} misses
 * combining marks; no UNICODE_CHARACTER_CLASS so \s stays ASCII — token-identical to
 * llama-tokenize, see tests/TokenizerParityTest). encode never maps special-token strings;
 * the chat format inserts special ids explicitly via {@link #getSpecialTokens()}.
 */
class LFMTokenizer {

    private static final String LFM2_PRE_PATTERN =
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)" +
                    "|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+" +
                    "|\\p{N}{1,3}" +
                    "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*" +
                    "|\\s*[\\r\\n]+" +
                    "|\\s+(?!\\S)" +
                    "|\\s+";

    private final Tokenizer tokenizer;
    private final Map<String, Integer> specialTokens;
    private Specials specialsEncoder; // lazy: only the --raw-prompt path needs it

    LFMTokenizer(GGUF gguf) {
        this.tokenizer = GGUFTokenizerLoader.createBuilderWithBuiltins()
                .registerPreTokenizer("lfm2", g -> Splitter.regex(Pattern.compile(LFM2_PRE_PATTERN)))
                .registerNormalizer("lfm2", g -> Normalizer.identity())
                .build()
                .fromGGUF(gguf);
        this.specialTokens = new HashMap<>();
        for (int id = 0; id < vocabularySize(); id++) {
            if (isSpecialToken(id)) {
                specialTokens.put(tokenizer.vocabulary().token(id), id);
            }
        }
    }

    public int vocabularySize() {
        return tokenizer.vocabulary().size();
    }

    public Map<String, Integer> getSpecialTokens() {
        return specialTokens;
    }

    public boolean isSpecialToken(int token) {
        return token >= 0 && token < vocabularySize()
                && !tokenizer.vocabulary().isTokenOfType(token, StandardTokenType.NORMAL);
    }

    List<Integer> encode(String text) {
        return tokenizer.encode(text).toList();
    }

    /** Encode mapping special-token strings in the text to their ids (plain {@link #encode}
     *  never maps them); the --raw-prompt path uses this to author templated streams as text. */
    List<Integer> encodeWithSpecialTokens(String text) {
        if (specialsEncoder == null) {
            specialsEncoder = Specials.compile(tokenizer.vocabulary(), specialTokens.keySet());
        }
        return specialsEncoder.encode(tokenizer, text).toList();
    }

    /** Raw UTF-8 bytes of one token (the streaming decoder assembles code points across tokens). */
    byte[] decodeTokenBytes(int token) {
        return tokenizer.decodeBytes(new int[]{token});
    }

    public String decode(int token) {
        return new String(decodeTokenBytes(token), StandardCharsets.UTF_8);
    }

    public String decode(List<Integer> tokens) {
        int[] ids = new int[tokens.size()];
        for (int i = 0; i < ids.length; i++) {
            ids[i] = tokens.get(i);
        }
        return new String(tokenizer.decodeBytes(ids), StandardCharsets.UTF_8);
    }
}
record Pair<First, Second>(First first, Second second) {
}

record GGMLTensorEntry(String name, GGMLType ggmlType, int[] shape,
                       MemorySegment memorySegment) {
}

