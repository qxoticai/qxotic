// Model I/O: GGUF parsing, tensor entries, vocabulary and the LFM tokenizer.
package com.llama4j;

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
import java.util.HexFormat;
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

final class GGUF {
    private static final int GGUF_MAGIC = 0x46554747;
    private static final int DEFAULT_ALIGNMENT = 32; // must be a power of 2
    private static final int PARSE_BUFFER_SIZE = 1 << 20;
    private int tensorCount;
    private int alignment;
    private Map<String, Object> metadata;
    private Map<String, GGUFTensorInfo> tensorInfos;
    private long tensorDataOffset;

    public Map<String, Object> getMetadata() {
        return metadata;
    }

    public long getTensorDataOffset() {
        return tensorDataOffset;
    }

    public Map<String, GGUFTensorInfo> getTensorInfos() {
        return tensorInfos;
    }

    private static final class ChannelReader {
        private final ReadableByteChannel channel;
        private final ByteBuffer buffer;
        private long position;

        private ChannelReader(ReadableByteChannel channel, int bufferSize) {
            this.channel = channel;
            this.buffer = ByteBuffer.allocateDirect(bufferSize).order(ByteOrder.LITTLE_ENDIAN);
            this.buffer.limit(0);
            this.position = 0L;
        }

        long position() {
            return position;
        }

        private void ensure(int required) throws IOException {
            if (required > buffer.capacity()) {
                throw new IllegalArgumentException("Requested read " + required + " exceeds buffer capacity " + buffer.capacity());
            }
            if (buffer.remaining() >= required) {
                return;
            }
            buffer.compact();
            while (buffer.position() < required) {
                int read = channel.read(buffer);
                if (read < 0) {
                    throw new IOException("Unexpected EOF while reading GGUF metadata");
                }
            }
            buffer.flip();
        }

        byte readByte() throws IOException {
            ensure(Byte.BYTES);
            position += Byte.BYTES;
            return buffer.get();
        }

        short readShort() throws IOException {
            ensure(Short.BYTES);
            position += Short.BYTES;
            return buffer.getShort();
        }

        int readInt() throws IOException {
            ensure(Integer.BYTES);
            position += Integer.BYTES;
            return buffer.getInt();
        }

        long readLong() throws IOException {
            ensure(Long.BYTES);
            position += Long.BYTES;
            return buffer.getLong();
        }

        float readFloat() throws IOException {
            return Float.intBitsToFloat(readInt());
        }

        double readDouble() throws IOException {
            return Double.longBitsToDouble(readLong());
        }

        byte[] readBytes(int length) throws IOException {
            byte[] bytes = new byte[length];
            int copied = 0;
            while (copied < length) {
                if (!buffer.hasRemaining()) {
                    ensure(1);
                }
                int chunk = Math.min(length - copied, buffer.remaining());
                buffer.get(bytes, copied, chunk);
                copied += chunk;
                position += chunk;
            }
            return bytes;
        }

        void skipBytes(int length) throws IOException {
            int remaining = length;
            while (remaining > 0) {
                if (!buffer.hasRemaining()) {
                    ensure(1);
                }
                int chunk = Math.min(remaining, buffer.remaining());
                buffer.position(buffer.position() + chunk);
                remaining -= chunk;
                position += chunk;
            }
        }
    }

    public static Map<String, GGMLTensorEntry> loadTensors(FileChannel fileChannel, long tensorDataOffset, Map<String, GGUFTensorInfo> tensorInfos) throws IOException {
        // Arena.global() is used intentionally: model weights are mapped for the lifetime of the process.
        // The OS will reclaim the mapping when the JVM exits; explicit cleanup is unnecessary.
        Arena arena = Arena.global();
        MemorySegment tensorData = fileChannel.map(FileChannel.MapMode.READ_ONLY, tensorDataOffset, fileChannel.size() - tensorDataOffset, arena);
        Map<String, GGMLTensorEntry> tensorEntries = HashMap.newHashMap(tensorInfos.size());
        for (Map.Entry<String, GGUFTensorInfo> entry : tensorInfos.entrySet()) {
            GGUFTensorInfo ti = entry.getValue();
            long numberOfElements = FloatTensor.numberOfElementsLong(ti.dimensions());
            long sizeInBytes = ti.ggmlType().byteSizeFor(numberOfElements);
            MemorySegment memorySegment = tensorData.asSlice(ti.offset(), sizeInBytes);
            tensorEntries.put(ti.name(), new GGMLTensorEntry(tensorData, ti.name(), ti.ggmlType(), ti.dimensions(), memorySegment));
        }
        return tensorEntries;
    }

    public static GGUF loadModel(Path modelPath) throws IOException {
        try (FileChannel fileChannel = FileChannel.open(modelPath)) {
            return loadModel(fileChannel, modelPath.toString());
        }
    }

    public static GGUF loadModel(FileChannel fileChannel, String modelLabel) throws IOException {
        try (var ignored = Timer.log("Parse " + modelLabel)) {
            fileChannel.position(0L);
            GGUF gguf = new GGUF();
            ChannelReader reader = new ChannelReader(fileChannel, PARSE_BUFFER_SIZE);
            gguf.loadModelImpl(reader);
            return gguf;
        }
    }

    enum MetadataValueType {
        UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL, STRING, ARRAY, UINT64, INT64, FLOAT64;

        private static final MetadataValueType[] VALUES = values();

        public static MetadataValueType fromIndex(int index) {
            return VALUES[index];
        }
    }

    private void loadModelImpl(ChannelReader reader) throws IOException {
        readHeader(reader);
        this.tensorInfos = HashMap.newHashMap(tensorCount);
        for (int i = 0; i < tensorCount; ++i) {
            GGUF.GGUFTensorInfo ti = readTensorInfo(reader);
            assert !tensorInfos.containsKey(ti.name);
            tensorInfos.put(ti.name, ti);
        }
        long position = reader.position();
        int padding = (int) ((getAlignment() - (position % getAlignment())) % getAlignment());
        skipBytes(reader, padding);
        this.tensorDataOffset = reader.position();
    }

    public record GGUFTensorInfo(String name, int[] dimensions, GGMLType ggmlType, long offset) {
    }

    private GGMLType readGGMLType(ChannelReader reader) throws IOException {
        int ggmlTypeId = readInt(reader);
        return GGMLType.fromId(ggmlTypeId);
    }

    private GGUF.GGUFTensorInfo readTensorInfo(ChannelReader reader) throws IOException {
        String name = readString(reader);
        assert name.length() <= 64;
        int n_dimensions = readInt(reader);
        assert n_dimensions <= 4;
        int[] dimensions = new int[n_dimensions];
        for (int i = 0; i < n_dimensions; ++i) {
            dimensions[i] = Math.toIntExact(readLong(reader));
        }
        GGMLType ggmlType = readGGMLType(reader);
        long offset = readLong(reader);
        assert offset % getAlignment() == 0;
        return new GGUF.GGUFTensorInfo(name, dimensions, ggmlType, offset);
    }

    private String readString(ChannelReader reader) throws IOException {
        int len = Math.toIntExact(readLong(reader));
        return new String(readBytes(reader, len), StandardCharsets.UTF_8);
    }

    private Pair<String, Object> readKeyValuePair(ChannelReader reader) throws IOException {
        String key = readString(reader);
        assert key.length() < (1 << 16);
        assert key.codePoints().allMatch(cp -> ('a' <= cp && cp <= 'z') || ('0' <= cp && cp <= '9') || cp == '_' || cp == '.');
        Object value = readMetadataValue(reader);
        return new Pair<>(key, value);
    }

    private Object readMetadataValue(ChannelReader reader) throws IOException {
        MetadataValueType valueType = readMetadataValueType(reader);
        return readMetadataValueOfType(valueType, reader);
    }

    void readHeader(ChannelReader reader) throws IOException {
        int magic = readInt(reader);
        if (magic != GGUF_MAGIC) {
            throw new IllegalArgumentException("unsupported header.magic " + magic);
        }
        int version = readInt(reader);
        if (version != 2 && version != 3) {
            throw new IllegalArgumentException("unsupported header.version " + version);
        }
        this.tensorCount = Math.toIntExact(readLong(reader));
        int metadata_kv_count = Math.toIntExact(readLong(reader));
        this.metadata = HashMap.newHashMap(metadata_kv_count);
        for (int i = 0; i < metadata_kv_count; ++i) {
            Pair<String, Object> keyValue = readKeyValuePair(reader);
            assert !metadata.containsKey(keyValue.first());
            metadata.put(keyValue.first(), keyValue.second());
        }
    }

    private Object readArray(ChannelReader reader) throws IOException {
        MetadataValueType valueType = readMetadataValueType(reader);
        int len = Math.toIntExact(readLong(reader));
        switch (valueType) {
            case UINT8, INT8 -> {
                return readBytes(reader, len);
            }
            case UINT16, INT16 -> {
                short[] shorts = new short[len];
                for (int i = 0; i < len; ++i) {
                    shorts[i] = readShort(reader);
                }
                return shorts;
            }
            case UINT32, INT32 -> {
                int[] ints = new int[len];
                for (int i = 0; i < len; ++i) {
                    ints[i] = readInt(reader);
                }
                return ints;
            }
            case FLOAT32 -> {
                float[] floats = new float[len];
                for (int i = 0; i < len; ++i) {
                    floats[i] = readFloat(reader);
                }
                return floats;
            }
            case BOOL -> {
                boolean[] booleans = new boolean[len];
                for (int i = 0; i < len; ++i) {
                    booleans[i] = readBoolean(reader);
                }
                return booleans;
            }
            case STRING -> {
                String[] strings = new String[len];
                for (int i = 0; i < len; ++i) {
                    strings[i] = readString(reader);
                }
                return strings;
            }
            case ARRAY -> {
                Object[] arrays = new Object[len];
                for (int i = 0; i < len; ++i) {
                    arrays[i] = readArray(reader);
                }
                return arrays;
            }
            default -> throw new UnsupportedOperationException("read array of " + valueType);
        }
    }

    private Object readMetadataValueOfType(MetadataValueType valueType, ChannelReader reader) throws IOException {
        return switch (valueType) {
            case UINT8, INT8 -> readByte(reader);
            case UINT16, INT16 -> readShort(reader);
            case UINT32, INT32 -> readInt(reader);
            case FLOAT32 -> readFloat(reader);
            case UINT64, INT64 -> readLong(reader);
            case FLOAT64 -> readDouble(reader);
            case BOOL -> readBoolean(reader);
            case STRING -> readString(reader);
            case ARRAY -> readArray(reader);
        };
    }

    private MetadataValueType readMetadataValueType(ChannelReader reader) throws IOException {
        int index = readInt(reader);
        return MetadataValueType.fromIndex(index);
    }

    private byte[] readBytes(ChannelReader reader, int length) throws IOException {
        return reader.readBytes(length);
    }

    private void skipBytes(ChannelReader reader, int length) throws IOException {
        reader.skipBytes(length);
    }

    private byte readByte(ChannelReader reader) throws IOException {
        return reader.readByte();
    }

    private boolean readBoolean(ChannelReader reader) throws IOException {
        return readByte(reader) != 0;
    }

    private short readShort(ChannelReader reader) throws IOException {
        return reader.readShort();
    }

    private int readInt(ChannelReader reader) throws IOException {
        return reader.readInt();
    }

    private long readLong(ChannelReader reader) throws IOException {
        return reader.readLong();
    }

    private float readFloat(ChannelReader reader) throws IOException {
        return reader.readFloat();
    }

    private double readDouble(ChannelReader reader) throws IOException {
        return reader.readDouble();
    }

    public int getAlignment() {
        if (alignment != 0) {
            return alignment;
        }
        alignment = (int) metadata.getOrDefault("general.alignment", DEFAULT_ALIGNMENT);
        assert Integer.bitCount(alignment) == 1 : "alignment must be a power of two";
        return alignment;
    }
}

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

    private static Vocabulary loadVocabulary(Map<String, Object> metadata) {
        String[] tokens = (String[]) metadata.get("tokenizer.ggml.tokens");
        float[] scores = (float[]) metadata.get("tokenizer.ggml.scores");
        return new Vocabulary(tokens, scores);
    }

    public static Llama loadModel(Path ggufPath, int contextLength) throws IOException {
        try (var ignored = Timer.log("Load LFM25 model")) {
            try (FileChannel fileChannel = FileChannel.open(ggufPath, StandardOpenOption.READ)) {
                GGUF gguf = GGUF.loadModel(fileChannel, ggufPath.toString());
                return loadModel(fileChannel, gguf, contextLength, true);
            }
        }
    }

    static Llama loadModel(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeightsFlag) throws IOException {
        Map<String, Object> metadata = gguf.getMetadata();

        Vocabulary vocabulary = loadVocabulary(metadata);
        LFMTokenizer tokenizer = createTokenizer(metadata, vocabulary);

        String archPrefix = (String) metadata.get("general.architecture");

        int modelContextLength = (int) metadata.get(archPrefix + ".context_length");
        if (contextLength < 0 || modelContextLength < contextLength) {
            contextLength = modelContextLength;
        }

        int embeddingLength = (int) metadata.get(archPrefix + ".embedding_length");
        int numberOfHeads = (int) metadata.get(archPrefix + ".attention.head_count");
        int numberOfLayers = (int) metadata.get(archPrefix + ".block_count");

        int headSizeFull = embeddingLength / numberOfHeads;
        int headSizeSWA = headSizeFull;
        int slidingWindow = 1;
        float logitSoftcapping = (float) metadata.getOrDefault(archPrefix + ".final_logit_softcapping", 0f);
        float rmsNormEps = (float) metadata.getOrDefault(archPrefix + ".attention.layer_norm_rms_epsilon", 1e-6f);
        float ropeTheta = (float) metadata.getOrDefault(archPrefix + ".rope.freq_base", 1000000f);
        float ropeThetaSWA = (float) metadata.getOrDefault(archPrefix + ".rope.freq_base_swa", 10000f);

        // MoE parameters
        int expertCount = (int) metadata.getOrDefault(archPrefix + ".expert_count", 0);
        int expertUsedCount = (int) metadata.getOrDefault(archPrefix + ".expert_used_count", 0);
        int expertFeedForwardLength = (int) metadata.getOrDefault(archPrefix + ".expert_feed_forward_length", 0);
        int leadingDenseBlockCount = (int) metadata.getOrDefault(archPrefix + ".leading_dense_block_count", numberOfLayers);
        int expertGatingFunc = (int) metadata.getOrDefault(archPrefix + ".expert_gating_func", 1); // 1=softmax, 2=sigmoid

        // Per-layer feed forward lengths (scalar for uniform, array for variable)
        int[] feedForwardLength;
        Object ffnRaw = metadata.get(archPrefix + ".feed_forward_length");
        if (ffnRaw instanceof int[] arr) {
            feedForwardLength = arr;
        } else {
            feedForwardLength = new int[numberOfLayers];
            Arrays.fill(feedForwardLength, (int) ffnRaw);
        }

        Map<String, GGUF.GGUFTensorInfo> tensorInfos = gguf.getTensorInfos();

        boolean[] isSWA = new boolean[numberOfLayers];

        int[] numberOfKeyValueHeadsPerLayer;
        Object kvHeads = metadata.get(archPrefix + ".attention.head_count_kv");
        if (kvHeads instanceof int[] arr && arr.length == numberOfLayers) {
            numberOfKeyValueHeadsPerLayer = arr;
        } else {
            numberOfKeyValueHeadsPerLayer = new int[numberOfLayers];
            for (int i = 0; i < numberOfLayers; i++) {
                GGUF.GGUFTensorInfo kWeight = tensorInfos.get("blk." + i + ".attn_k.weight");
                numberOfKeyValueHeadsPerLayer[i] = kWeight != null ? kWeight.dimensions()[1] / headSizeFull : 0;
            }
        }
        int nLayerKvFromStart = numberOfLayers;
        int shortConvLCache = (int) metadata.getOrDefault(archPrefix + ".shortconv.l_cache", 0);


        Llama.Configuration config = new Llama.Configuration(
                embeddingLength,
                feedForwardLength,
                numberOfLayers,
                numberOfHeads,
                numberOfKeyValueHeadsPerLayer,
                vocabulary.size(),
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

        Map<String, GGMLTensorEntry> tensorEntries = GGUF.loadTensors(fileChannel, gguf.getTensorDataOffset(), tensorInfos);
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

    private static LFMTokenizer createTokenizer(Map<String, Object> metadata, Vocabulary vocabulary) {
        int[] tokenTypes = (int[]) metadata.get("tokenizer.ggml.token_type");
        String[] merges = (String[]) metadata.get("tokenizer.ggml.merges");
        return new LFMTokenizer(vocabulary, merges, tokenTypes);
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

class LFMTokenizer {
    private static final String LLAMA3_PATTERN =
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)" +
                    "|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+" +
                    "|\\p{N}{1,3}" +
                    "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*" +
                    "|\\s*[\\r\\n]+" +
                    "|\\s+(?!\\S)" +
                    "|\\s+";

    private final Pattern compiledPattern;
    private final Vocabulary vocabulary;
    private final Map<Pair<Integer, Integer>, Integer> merges;
    private final Map<String, Integer> specialTokens;
    private final BitSet specialTokenMask;

    public LFMTokenizer(Vocabulary vocabulary, String[] mergeLines, int[] tokenType) {
        this.vocabulary = vocabulary;
        this.compiledPattern = Pattern.compile(LLAMA3_PATTERN);
        this.merges = HashMap.newHashMap(mergeLines.length);
        for (Pair<Integer, Integer> pair : Arrays.stream(mergeLines)
                .map(line -> line.split(" "))
                .map(parts -> new Pair<>(
                        vocabulary.getIndex(parts[0]).orElseThrow(),
                        vocabulary.getIndex(parts[1]).orElseThrow()))
                .toList()) {
            int mergeIndex = vocabulary.getIndex(vocabulary.get(pair.first()) + vocabulary.get(pair.second())).orElseThrow();
            this.merges.put(pair, mergeIndex);
        }
        this.specialTokens = new HashMap<>();
        this.specialTokenMask = new BitSet(tokenType.length);
        for (int i = 0; i < tokenType.length; i++) {
            if (tokenType[i] != 1) {
                specialTokens.put(vocabulary.get(i), i);
                specialTokenMask.set(i);
            }
        }
    }

    public Map<String, Integer> getSpecialTokens() { return specialTokens; }
    public boolean isSpecialToken(int tokenIndex) { return tokenIndex >= 0 && specialTokenMask.get(tokenIndex); }

    /** Decode a single token without List allocation overhead. */
    public String decode(int token) {
        return decode(List.of(token));
    }

    private static Map<Integer, Integer> bytesToUnicode() {
        List<Integer> bs = new ArrayList<>();
        for (int c = '!'; c <= '~'; c++) bs.add(c);
        for (int c = 0x00A1; c <= 0x00AC; c++) bs.add(c);
        for (int c = 0x00AE; c <= 0x00FF; c++) bs.add(c);
        List<Integer> cs = new ArrayList<>(bs);
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            if (!bs.contains(b)) {
                bs.add(b);
                cs.add(256 + n);
                n += 1;
            }
        }
        Map<Integer, Integer> result = HashMap.newHashMap(bs.size());
        for (int i = 0; i < bs.size(); i++) {
            result.put(bs.get(i), cs.get(i));
        }
        return result;
    }

    static final Map<Integer, Integer> BYTE_ENCODER = bytesToUnicode();
    static final Map<Integer, Integer> BYTE_DECODER = BYTE_ENCODER.entrySet().stream().collect(Collectors.toMap(Map.Entry::getValue, Map.Entry::getKey));

    List<Integer> encode(String text) {
        StringBuilder sb = new StringBuilder();
        byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
        for (byte b : bytes) {
            sb.appendCodePoint(BYTE_ENCODER.get(Byte.toUnsignedInt(b)));
        }
        return encodeOrdinary(sb.toString());
    }

    private List<Integer> encodeOrdinary(String text) {
        List<String> chunks = findAll(compiledPattern, text);
        List<Integer> ids = new ArrayList<>();
        for (String chunk : chunks) {
            ids.addAll(encodeChunk(chunk));
        }
        return ids;
    }

    private static List<String> findAll(Pattern pattern, String text) {
        List<String> allMatches = new ArrayList<>();
        Matcher matcher = pattern.matcher(text);
        while (matcher.find()) {
            allMatches.add(matcher.group());
        }
        return allMatches;
    }

    private List<Integer> encodeChunk(String chunk) {
        List<Integer> ids = new ArrayList<>();
        for (int b : chunk.toCharArray()) {
            ids.add(vocabulary.getIndex(String.valueOf((char) b)).orElseThrow());
        }
        while (ids.size() >= 2) {
            Pair<Integer, Integer> pair = getStats(ids).keySet().stream()
                    .min(Comparator.comparingInt(key -> this.merges.getOrDefault(key, Integer.MAX_VALUE)))
                    .orElseThrow();
            if (!this.merges.containsKey(pair)) {
                break;
            }
            ids = merge(ids, pair, this.merges.get(pair));
        }
        return ids;
    }

    private Map<Pair<Integer, Integer>, Integer> getStats(List<Integer> ids) {
        Map<Pair<Integer, Integer>, Integer> map = new HashMap<>();
        for (int i = 0; i + 1 < ids.size(); i++) {
            Pair<Integer, Integer> key = new Pair<>(ids.get(i), ids.get(i + 1));
            map.put(key, map.getOrDefault(key, 0) + 1);
        }
        return map;
    }

    private static List<Integer> merge(List<Integer> ids, Pair<Integer, Integer> pair, int idx) {
        List<Integer> merged = new ArrayList<>();
        int i = 0;
        while (i < ids.size()) {
            if (ids.get(i).equals(pair.first()) && i < ids.size() - 1 && ids.get(i + 1).equals(pair.second())) {
                merged.add(idx);
                i += 2;
            } else {
                merged.add(ids.get(i));
                i += 1;
            }
        }
        return merged;
    }

    public String decode(List<Integer> tokens) {
        StringBuilder result = new StringBuilder();
        List<Integer> pending = new ArrayList<>();
        for (int token : tokens) {
            if (isSpecialToken(token)) {
                if (!pending.isEmpty()) {
                    result.append(decodeBPE(pending));
                    pending.clear();
                }
                result.append(vocabulary.get(token));
            } else {
                pending.add(token);
            }
        }
        if (!pending.isEmpty()) {
            result.append(decodeBPE(pending));
        }
        return result.toString();
    }

    private String decodeBPE(List<Integer> tokens) {
        StringBuilder sb = new StringBuilder();
        for (int token : tokens) sb.append(vocabulary.get(token));
        String decoded = sb.toString();
        int[] ints = decoded.codePoints().map(BYTE_DECODER::get).toArray();
        byte[] raw = new byte[ints.length];
        for (int i = 0; i < raw.length; i++) raw[i] = (byte) ints[i];
        return new String(raw, StandardCharsets.UTF_8);
    }

    byte[] decodeTokenBytes(int token) {
        if (isSpecialToken(token)) {
            return vocabulary.get(token).getBytes(StandardCharsets.UTF_8);
        }
        int[] cps = vocabulary.get(token).codePoints().toArray();
        byte[] out = new byte[cps.length];
        for (int i = 0; i < cps.length; i++) {
            int b = BYTE_DECODER.get(cps[i]);
            out[i] = (byte) b;
        }
        return out;
    }

    public static String replaceControlCharacters(int[] codePoints) {
        StringBuilder chars = new StringBuilder();
        for (int cp : codePoints) {
            if (Character.getType(cp) == Character.CONTROL && cp != '\n') {
                chars.append("\\u").append(HexFormat.of().toHexDigits(cp, 4));
            } else {
                chars.appendCodePoint(cp);
            }
        }
        return chars.toString();
    }

    public static String replaceControlCharacters(String str) {
        return replaceControlCharacters(str.codePoints().toArray());
    }

}

record Pair<First, Second>(First first, Second second) {
}

record GGMLTensorEntry(MemorySegment mappedFile, String name, GGMLType ggmlType, int[] shape,
                       MemorySegment memorySegment) {
}

record Vocabulary(String[] tokens, float[] scores, Map<String, Integer> tokenToIndex) {
    public Vocabulary(String[] vocabulary, float[] scores) {
        this(vocabulary, scores,
                IntStream.range(0, vocabulary.length)
                        .boxed()
                        .collect(Collectors.toMap(i -> vocabulary[i], i -> i))
        );
    }

    public String get(int tokenIndex) {
        return tokens[tokenIndex];
    }

    public OptionalInt getIndex(String token) {
        Integer value = tokenToIndex.get(token);
        return value != null ? OptionalInt.of(value) : OptionalInt.empty();
    }

    public int size() {
        return tokens.length;
    }

    public float getScore(int tokenIndex) {
        return scores[tokenIndex];
    }
}
