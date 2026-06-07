///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 21+
//PREVIEW
//COMPILE_OPTIONS --add-modules=jdk.incubator.vector
//RUNTIME_OPTIONS --add-modules=jdk.incubator.vector -Djdk.incubator.vector.VECTOR_ACCESS_OOB_CHECK=0
//MAIN com.llama4j.LFM25

// LFM2.5 inference in pure Java
// Author: Alfonso² Peterssen
// Based on Andrej Karpathy's llama2.c and minbpe projects
// Related project: https://github.com/mukel/llama3.java
//
// Supports GGUF models and multiple tensor formats
// Matrix-vector kernels use Java's Vector API
// CLI modes: --chat and --instruct
//
// Run:
// jbang LFM25.java --help
package com.llama4j;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorShape;
import jdk.incubator.vector.VectorSpecies;

import java.lang.reflect.Field;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
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
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.OptionalInt;
import java.util.Set;
import java.util.concurrent.TimeUnit;
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

        int embeddingLengthPerLayer = (int) metadata.getOrDefault(archPrefix + ".embedding_length_per_layer_input", 0);

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
                embeddingLengthPerLayer,
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
            FloatBuffer ropeFreqsBuf = toFloatBuffer(ropeFreqEntry);
            float[] modelRopeFreqs = new float[ropeFreqsBuf.remaining()];
            ropeFreqsBuf.get(modelRopeFreqs);
            ropeFreqsFull = RoPE.precomputeFreqsCisFromFreqs(config.contextLength, config.headSizeFull, config.ropeTheta, modelRopeFreqs);
        } else {
            ropeFreqsFull = RoPE.precomputeFreqsCis(config.contextLength, config.headSizeFull, config.ropeTheta);
        }
        return loadWeightsWithRoPE(tensorEntries, config, ropeFreqsSWA, ropeFreqsFull);
    }

    public static Llama.Weights loadWeightsWithRoPE(Map<String, GGMLTensorEntry> tensorEntries, Llama.Configuration config,
                                                     Pair<float[], float[]> ropeFreqsSWA, Pair<float[], float[]> ropeFreqsFull) {
        int numberOfLayers = config.numberOfLayers;

        FloatTensor tokenEmbeddingTable = loadQuantized(tensorEntries.get("token_embd.weight"));

        // Load per-layer output scale (scalar per layer)
        float[] layerOutputScale = new float[config.numberOfLayers];
        for (int i = 0; i < config.numberOfLayers; i++) {
            GGMLTensorEntry scaleEntry = tensorEntries.get("blk." + i + ".layer_output_scale.weight");
            if (scaleEntry != null) {
                layerOutputScale[i] = toFloatBuffer(scaleEntry).get(0);
            } else {
                layerOutputScale[i] = 1.0f;
            }
        }

        // Load per-layer embedding weights (if present)
        FloatTensor perLayerTokenEmbd = null;
        FloatTensor perLayerModelProj = null;
        FloatBuffer perLayerProjNorm = null;
        FloatTensor[] perLayerInpGate = null;
        FloatTensor[] perLayerProj = null;
        FloatBuffer[] perLayerPostNorm = null;

        if (config.embeddingLengthPerLayer > 0 && tensorEntries.containsKey("per_layer_token_embd.weight")) {
            perLayerTokenEmbd = loadQuantized(tensorEntries.get("per_layer_token_embd.weight"));
            perLayerModelProj = loadQuantized(tensorEntries.get("per_layer_model_proj.weight"));
            perLayerProjNorm = toFloatBuffer(tensorEntries.get("per_layer_proj_norm.weight"));
            perLayerInpGate = loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".inp_gate.weight"));
            perLayerProj = loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".proj.weight"));
            perLayerPostNorm = loadArrayOfFloatBuffer(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".post_norm.weight"));
        }

        // Load V weights (nullable: layers without V use K as V)
        FloatTensor[] wv = new FloatTensor[numberOfLayers];
        for (int i = 0; i < numberOfLayers; i++) {
            GGMLTensorEntry vEntry = tensorEntries.get("blk." + i + ".attn_v.weight");
            wv[i] = vEntry != null ? loadQuantized(vEntry) : null;
        }

        // Load MoE weights (if present)
        FloatTensor[] ffnGateInp = null;
        FloatTensor[] ffnGateExps = null;
        FloatTensor[] ffnUpExps = null;
        FloatTensor[] ffnDownExps = null;
        FloatBuffer[] ffnExpProbsB = null;

        if (config.isMoE()) {
            ffnGateInp = loadArrayOfQuantized(numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_gate_inp.weight"));
            ffnGateExps = loadArrayOfQuantized(numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_gate_exps.weight"));
            ffnUpExps = loadArrayOfQuantized(numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_up_exps.weight"));
            ffnDownExps = loadArrayOfQuantized(numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_down_exps.weight"));
            ffnExpProbsB = loadArrayOfFloatBuffer(numberOfLayers, i -> tensorEntries.get("blk." + i + ".exp_probs_b.bias"));
        }

        return new Llama.Weights(
                tokenEmbeddingTable,
                loadArrayOfFloatBuffer(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_norm.weight")),
                loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_q.weight")),
                loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_k.weight")),
                wv,
                loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_output.weight")),
                loadArrayOfFloatBuffer(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_q_norm.weight")),
                loadArrayOfFloatBuffer(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_k_norm.weight")),
                loadArrayOfFloatBuffer(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".post_attention_norm.weight")),
                loadArrayOfFloatBuffer(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_norm.weight")),
                loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_gate.weight")),
                loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_down.weight")),
                loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_up.weight")),
                loadArrayOfFloatBuffer(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".post_ffw_norm.weight")),
                toFloatBuffer(tensorEntries.getOrDefault("output_norm.weight", tensorEntries.get("token_embd_norm.weight"))),
                layerOutputScale,
                FloatBuffer.wrap(ropeFreqsFull.first()),
                FloatBuffer.wrap(ropeFreqsFull.second()),
                FloatBuffer.wrap(ropeFreqsSWA.first()),
                FloatBuffer.wrap(ropeFreqsSWA.second()),
                tensorEntries.containsKey("output.weight")
                        ? loadQuantized(tensorEntries.get("output.weight"))
                        : tokenEmbeddingTable,
                perLayerTokenEmbd, perLayerModelProj, perLayerProjNorm,
                perLayerInpGate, perLayerProj, perLayerPostNorm,
                ffnGateInp, ffnGateExps, ffnUpExps, ffnDownExps, ffnExpProbsB,
                loadArrayOfFloatBuffer(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".shortconv.conv.weight")),
                loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".shortconv.in_proj.weight")),
                loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".shortconv.out_proj.weight"))
        );
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

    public static FloatTensor[] loadArrayOfQuantized(int size, IntFunction<GGMLTensorEntry> getTensorEntry) {
        FloatTensor[] array = new FloatTensor[size];
        for (int i = 0; i < size; i++) {
            GGMLTensorEntry entry = getTensorEntry.apply(i);
            array[i] = entry != null ? loadQuantized(entry) : null;
        }
        return array;
    }

    public static FloatBuffer[] loadArrayOfFloatBuffer(int size, IntFunction<GGMLTensorEntry> getTensorEntry) {
        FloatBuffer[] array = new FloatBuffer[size];
        for (int i = 0; i < size; i++) {
            GGMLTensorEntry entry = getTensorEntry.apply(i);
            array[i] = entry != null ? toFloatBuffer(entry) : null;
        }
        return array;
    }

    public static FloatBuffer toFloatBuffer(GGMLTensorEntry tensorEntry) {
        GGMLType ggmlType = tensorEntry.ggmlType();
        return switch (ggmlType) {
            case F32 -> tensorEntry.memorySegment().asByteBuffer().order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
            default -> throw new UnsupportedOperationException("Conversion to " + ggmlType);
        };
    }
}

record Llama(Configuration configuration, LFMTokenizer tokenizer, Weights weights) {
    public State createNewState() {
        State state = new State(configuration());
        Integer bos = tokenizer.getSpecialTokens().get("<bos>");
        if (bos == null) bos = tokenizer.getSpecialTokens().get("<|startoftext|>");
        if (bos == null) bos = 1;
        state.latestToken = bos;
        return state;
    }

    public static final class Configuration {
        public final int embeddingLength;
        public final int[] feedForwardLength; // per-layer (shared MLP)
        public final int numberOfLayers;
        public final int numberOfHeads;
        public final int[] numberOfKeyValueHeadsPerLayer; // per-layer KV head count
        public final int vocabularySize;
        public final int contextLength;
        public final float rmsNormEps;
        public final float ropeTheta;       // full attention RoPE theta
        public final float ropeThetaSWA;    // SWA RoPE theta
        public final int headSizeFull;      // head size for full attention layers
        public final int headSizeSWA;       // head size for SWA layers
        public final int slidingWindow;
        public final float logitSoftcapping;
        public final boolean[] isSWA;       // per-layer: true = SWA, false = full attention
        public final int nLayerKvFromStart; // first N layers have own KV cache, rest reuse
        public final int embeddingLengthPerLayer; // per-layer embedding dim (0 = disabled)
        // MoE fields
        public final int expertCount;           // 0 = dense model (no MoE)
        public final int expertUsedCount;       // top-k experts per token
        public final int expertFeedForwardLength; // expert FFN hidden dim
        public final int shortConvLCache;
        public final int leadingDenseBlockCount; // first N layers use dense FFN (rest use MoE)
        public final int expertGatingFunc;      // 1=softmax, 2=sigmoid

        public Configuration(int embeddingLength, int[] feedForwardLength, int numberOfLayers,
                             int numberOfHeads, int[] numberOfKeyValueHeadsPerLayer, int vocabularySize,
                             int contextLength, float rmsNormEps, float ropeTheta, float ropeThetaSWA,
                             int headSizeFull, int headSizeSWA, int slidingWindow,
                             float logitSoftcapping, boolean[] isSWA, int nLayerKvFromStart,
                             int embeddingLengthPerLayer,
                             int expertCount, int expertUsedCount, int expertFeedForwardLength,
                             int shortConvLCache,
                             int leadingDenseBlockCount,
                             int expertGatingFunc) {
            if (slidingWindow <= 0 || Integer.bitCount(slidingWindow) != 1) {
                throw new IllegalArgumentException("slidingWindow must be a power of 2, got " + slidingWindow);
            }
            this.embeddingLength = embeddingLength;
            this.feedForwardLength = feedForwardLength;
            this.numberOfLayers = numberOfLayers;
            this.numberOfHeads = numberOfHeads;
            this.numberOfKeyValueHeadsPerLayer = numberOfKeyValueHeadsPerLayer;
            this.vocabularySize = vocabularySize;
            this.contextLength = contextLength;
            this.rmsNormEps = rmsNormEps;
            this.ropeTheta = ropeTheta;
            this.ropeThetaSWA = ropeThetaSWA;
            this.headSizeFull = headSizeFull;
            this.headSizeSWA = headSizeSWA;
            this.slidingWindow = slidingWindow;
            this.logitSoftcapping = logitSoftcapping;
            this.isSWA = isSWA;
            this.nLayerKvFromStart = nLayerKvFromStart;
            this.embeddingLengthPerLayer = embeddingLengthPerLayer;
            this.expertCount = expertCount;
            this.expertUsedCount = expertUsedCount;
            this.expertFeedForwardLength = expertFeedForwardLength;
            this.shortConvLCache = shortConvLCache;
            this.leadingDenseBlockCount = leadingDenseBlockCount;
            this.expertGatingFunc = expertGatingFunc;
        }

        public boolean isMoE() { return expertCount > 0; }
        public boolean isMoELayer(int layer) { return expertCount > 0 && layer >= leadingDenseBlockCount; }

        // For layers without own KV, return the layer whose cache to reuse
        public int kvSourceLayer(int layer) {
            if (layer < nLayerKvFromStart) return layer; // has own KV
            // Reuse the last KV layer of the same attention type
            return nLayerKvFromStart - (isSWA[layer] ? 2 : 1);
        }

        public boolean hasKv(int layer) {
            return layer < nLayerKvFromStart;
        }

        public int headSize(int layer) {
            return isSWA[layer] ? headSizeSWA : headSizeFull;
        }

        public int numberOfKeyValueHeads(int layer) {
            return numberOfKeyValueHeadsPerLayer[layer];
        }

        public int kvCachePositions(int layer) {
            return isSWA[layer] ? Math.min(contextLength, slidingWindow) : contextLength;
        }

        public int kvCacheIndex(int layer, int position) {
            return isSWA[layer] ? (position & (slidingWindow - 1)) : position;
        }

        public int kvDim(int layer) {
            return numberOfKeyValueHeadsPerLayer[layer] * headSize(layer);
        }

        public int queryDim(int layer) {
            return numberOfHeads * headSize(layer);
        }

        public boolean isRecurrentLayer(int layer) {
            return numberOfKeyValueHeadsPerLayer[layer] == 0;
        }

        public int maxHiddenDim() {
            return Arrays.stream(feedForwardLength).max().orElseThrow();
        }

        public Configuration withContextLength(int newContextLength) {
            return new Configuration(embeddingLength, feedForwardLength, numberOfLayers,
                    numberOfHeads, numberOfKeyValueHeadsPerLayer, vocabularySize,
                    newContextLength, rmsNormEps, ropeTheta, ropeThetaSWA,
                    headSizeFull, headSizeSWA, slidingWindow,
                    logitSoftcapping, isSWA, nLayerKvFromStart,
                    embeddingLengthPerLayer,
                    expertCount, expertUsedCount, expertFeedForwardLength,
                    shortConvLCache,
                    leadingDenseBlockCount,
                    expertGatingFunc);
        }
    }

    public static final class Weights {
        public final FloatTensor token_embedding_table;
        public final FloatBuffer[] rms_att_weight;       // (layer, dim)
        public final FloatTensor[] wq;                   // (layer, queryDim, dim)
        public final FloatTensor[] wk;                   // (layer, kvDim, dim)
        public final FloatTensor[] wv;                   // (layer, kvDim, dim) - null entry if V=K
        public final FloatTensor[] wo;                   // (layer, dim, queryDim)
        public final FloatBuffer[] attn_q_norm;          // (layer, headSize)
        public final FloatBuffer[] attn_k_norm;          // (layer, headSize)
        public final FloatBuffer[] post_attention_norm;  // (layer, dim)
        public final FloatBuffer[] rms_ffn_weight;       // (layer, dim) - shared MLP norm
        public final FloatTensor[] w1;                   // ffn_gate (layer, hiddenDim, dim)
        public final FloatTensor[] w2;                   // ffn_down (layer, dim, hiddenDim)
        public final FloatTensor[] w3;                   // ffn_up (layer, hiddenDim, dim)
        public final FloatBuffer[] post_ffw_norm;        // (layer, dim) - overall post-FFW norm
        public final FloatBuffer rms_final_weight;
        public final float[] layerOutputScale;
        // Full attention RoPE
        public final FloatBuffer freq_cis_real_full;
        public final FloatBuffer freq_cis_imag_full;
        // SWA RoPE
        public final FloatBuffer freq_cis_real_swa;
        public final FloatBuffer freq_cis_imag_swa;
        public final FloatTensor wcls;
        // Per-layer embedding weights
        public final FloatTensor perLayerTokenEmbd;
        public final FloatTensor perLayerModelProj;
        public final FloatBuffer perLayerProjNorm;
        public final FloatTensor[] perLayerInpGate;
        public final FloatTensor[] perLayerProj;
        public final FloatBuffer[] perLayerPostNorm;
        // MoE weights (null if dense model or for dense layers)
        public final FloatTensor[] ffnGateInp;        // router weight (layer, n_experts, n_embd)
        public final FloatTensor[] ffnGateExps;       // expert gate weights (layer, n_experts * expert_ff, n_embd)
        public final FloatTensor[] ffnUpExps;         // expert up weights (layer, n_experts * expert_ff, n_embd)
        public final FloatTensor[] ffnDownExps;       // down expert (layer, n_experts * n_embd, expert_ff)
        public final FloatBuffer[] ffnExpProbsB;      // expert probability bias (layer, n_experts)
        public final FloatBuffer[] shortConvKernel;    // (layer, k, dim)
        public final FloatTensor[] shortConvInProj;    // (layer, 3*dim, dim)
        public final FloatTensor[] shortConvOutProj;   // (layer, dim, dim)

        public Weights(FloatTensor token_embedding_table,
                       FloatBuffer[] rms_att_weight,
                       FloatTensor[] wq, FloatTensor[] wk, FloatTensor[] wv, FloatTensor[] wo,
                       FloatBuffer[] attn_q_norm, FloatBuffer[] attn_k_norm,
                       FloatBuffer[] post_attention_norm,
                       FloatBuffer[] rms_ffn_weight,
                       FloatTensor[] w1, FloatTensor[] w2, FloatTensor[] w3,
                       FloatBuffer[] post_ffw_norm,
                       FloatBuffer rms_final_weight,
                       float[] layerOutputScale,
                       FloatBuffer freq_cis_real_full, FloatBuffer freq_cis_imag_full,
                       FloatBuffer freq_cis_real_swa, FloatBuffer freq_cis_imag_swa,
                       FloatTensor wcls,
                       FloatTensor perLayerTokenEmbd, FloatTensor perLayerModelProj,
                       FloatBuffer perLayerProjNorm,
                       FloatTensor[] perLayerInpGate, FloatTensor[] perLayerProj,
                       FloatBuffer[] perLayerPostNorm,
                        FloatTensor[] ffnGateInp,
                        FloatTensor[] ffnGateExps, FloatTensor[] ffnUpExps, FloatTensor[] ffnDownExps,
                        FloatBuffer[] ffnExpProbsB,
                        FloatBuffer[] shortConvKernel, FloatTensor[] shortConvInProj,
                        FloatTensor[] shortConvOutProj) {
            this.token_embedding_table = token_embedding_table;
            this.rms_att_weight = rms_att_weight;
            this.wq = wq;
            this.wk = wk;
            this.wv = wv;
            this.wo = wo;
            this.attn_q_norm = attn_q_norm;
            this.attn_k_norm = attn_k_norm;
            this.post_attention_norm = post_attention_norm;
            this.rms_ffn_weight = rms_ffn_weight;
            this.w1 = w1;
            this.w2 = w2;
            this.w3 = w3;
            this.post_ffw_norm = post_ffw_norm;
            this.rms_final_weight = rms_final_weight;
            this.layerOutputScale = layerOutputScale;
            this.freq_cis_real_full = freq_cis_real_full;
            this.freq_cis_imag_full = freq_cis_imag_full;
            this.freq_cis_real_swa = freq_cis_real_swa;
            this.freq_cis_imag_swa = freq_cis_imag_swa;
            this.wcls = wcls;
            this.perLayerTokenEmbd = perLayerTokenEmbd;
            this.perLayerModelProj = perLayerModelProj;
            this.perLayerProjNorm = perLayerProjNorm;
            this.perLayerInpGate = perLayerInpGate;
            this.perLayerProj = perLayerProj;
            this.perLayerPostNorm = perLayerPostNorm;
            this.ffnGateInp = ffnGateInp;
            this.ffnGateExps = ffnGateExps;
            this.ffnUpExps = ffnUpExps;
            this.ffnDownExps = ffnDownExps;
            this.ffnExpProbsB = ffnExpProbsB;
            this.shortConvKernel = shortConvKernel;
            this.shortConvInProj = shortConvInProj;
            this.shortConvOutProj = shortConvOutProj;
        }
    }

    public static final class State {
        public final FloatTensor x;      // activation at current time stamp (embeddingLength,)
        public final FloatTensor xb;     // same, but inside a residual branch (embeddingLength,)
        public final FloatTensor xb_k;   // attention output before wo projection (max queryDim,)
        public final FloatTensor xb2;    // an additional buffer (embeddingLength,)
        public final FloatTensor hb;     // buffer for hidden dimension in the ffn (maxHiddenDim,)
        public final FloatTensor hb2;    // buffer for hidden dimension in the ffn (maxHiddenDim,)
        public final FloatTensor q;      // query (max queryDim,)
        public final FloatTensor k;      // key (max kvDim,)
        public final FloatTensor v;      // value (max kvDim,)
        public final FloatTensor att;    // buffer for scores/attention values (n_heads, seq_len)
        public final FloatTensor logits; // output logits
        // kv cache - variable sizes per layer
        public final FloatTensor[] keyCache;   // (n_layer, seq_len, kvDim_per_layer)
        public final FloatTensor[] valueCache; // (n_layer, seq_len, kvDim_per_layer)
        // per-layer embedding buffers
        public final FloatTensor perLayerInputs;
        public final FloatTensor plGate;
        public final FloatTensor plProj;
        // MoE buffers
        public final FloatTensor routerLogits;    // (n_experts,)
        public final FloatTensor expertGate;      // (expert_ff,)
        public final FloatTensor expertUp;        // (expert_ff,)
        public final FloatTensor expertDown;      // (n_embd,) single expert output
        public final int[] topExperts;            // reusable MoE top-k indexes
        public final float[] topProbs;            // reusable MoE top-k weights
        public final FloatTensor[] shortConvState; // recurrent layer -> (d_conv - 1, dim)
        public final FloatTensor shortConvTmp;     // (3*dim,)

        public int latestToken;

        State(Configuration config) {
            int maxQueryDim = config.numberOfHeads * config.headSizeFull;
            int maxKVDim = IntStream.range(0, config.numberOfLayers).map(config::kvDim).max().orElse(0);
            int maxHiddenDim = config.maxHiddenDim();
            this.x = ArrayFloatTensor.allocate(config.embeddingLength);
            this.xb = ArrayFloatTensor.allocate(config.embeddingLength);
            this.xb_k = ArrayFloatTensor.allocate(maxQueryDim);
            this.xb2 = ArrayFloatTensor.allocate(config.embeddingLength);
            this.hb = ArrayFloatTensor.allocate(maxHiddenDim);
            this.hb2 = ArrayFloatTensor.allocate(maxHiddenDim);
            this.q = ArrayFloatTensor.allocate(maxQueryDim);
            this.k = ArrayFloatTensor.allocate(maxKVDim);
            this.v = ArrayFloatTensor.allocate(maxKVDim);
            this.att = ArrayFloatTensor.allocate(config.numberOfHeads, config.contextLength);
            this.logits = ArrayFloatTensor.allocate(config.vocabularySize);
            this.shortConvTmp = ArrayFloatTensor.allocate(3 * config.embeddingLength);
            int plDim = config.embeddingLengthPerLayer;
            this.perLayerInputs = plDim > 0 ? ArrayFloatTensor.allocate(plDim * config.numberOfLayers) : null;
            this.plGate = plDim > 0 ? ArrayFloatTensor.allocate(plDim) : null;
            this.plProj = plDim > 0 ? ArrayFloatTensor.allocate(config.embeddingLength) : null;
            // MoE buffers
            if (config.isMoE()) {
                this.routerLogits = ArrayFloatTensor.allocate(config.expertCount);
                this.expertGate = ArrayFloatTensor.allocate(config.expertFeedForwardLength);
                this.expertUp = ArrayFloatTensor.allocate(config.expertFeedForwardLength);
                this.expertDown = ArrayFloatTensor.allocate(config.embeddingLength);
                this.topExperts = new int[config.expertUsedCount];
                this.topProbs = new float[config.expertUsedCount];
            } else {
                this.routerLogits = null;
                this.expertGate = null;
                this.expertUp = null;
                this.expertDown = null;
                this.topExperts = null;
                this.topProbs = null;
            }
            // Only allocate KV caches for layers that have their own KV (not shared)
            this.keyCache = new FloatTensor[config.nLayerKvFromStart];
            this.valueCache = new FloatTensor[config.nLayerKvFromStart];
            for (int l = 0; l < config.nLayerKvFromStart; l++) {
                int kvDim = config.kvDim(l);
                int kvPositions = config.kvCachePositions(l);
                keyCache[l] = F16FloatTensor.allocate(kvPositions, kvDim);
                valueCache[l] = F16FloatTensor.allocate(kvPositions, kvDim);
            }
            this.shortConvState = new FloatTensor[config.numberOfLayers];
            int dConv = Math.max(config.shortConvLCache - 1, 0);
            if (dConv > 0) {
                for (int l = 0; l < config.numberOfLayers; l++) {
                    if (config.isRecurrentLayer(l)) {
                        shortConvState[l] = ArrayFloatTensor.allocate(dConv * config.embeddingLength);
                    }
                }
            }
        }
    }

    static float gelu(float x) {
        return (float) (0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * Math.pow(x, 3)))));
    }

    static float silu(float x) {
        return (float) (x / (1.0 + Math.exp(-x)));
    }

    static void rmsnorm(FloatTensor out, FloatTensor x, FloatBuffer weight, int size, float rmsNormEps) {
        float ss = x.reduce(0, size, 0f, (acc, xi) -> acc + xi * xi);
        ss /= size;
        ss += rmsNormEps;
        ss = (float) (1.0 / Math.sqrt(ss));
        final float finalss = ss;
        out.mapWithIndexInPlace(0, size, (value, index) -> weight.get(index) * (finalss * x.getFloat(index)));
    }

    static void rmsnorm(FloatTensor out, int outOffset, FloatTensor x, int xOffset, FloatBuffer weight, int size, float rmsNormEps) {
        float ss = 0f;
        for (int i = 0; i < size; i++) {
            float xi = x.getFloat(xOffset + i);
            ss += xi * xi;
        }
        ss /= size;
        ss += rmsNormEps;
        ss = (float) (1.0 / Math.sqrt(ss));
        for (int i = 0; i < size; i++) {
            out.setFloat(outOffset + i, weight.get(i) * ss * x.getFloat(xOffset + i));
        }
    }

    // Bare RMS norm without learned weights (just normalize to unit RMS)
    static void rmsnormNoWeight(FloatTensor out, int outOffset, FloatTensor x, int xOffset, int size, float rmsNormEps) {
        float ss = 0f;
        for (int i = 0; i < size; i++) {
            float xi = x.getFloat(xOffset + i);
            ss += xi * xi;
        }
        ss /= size;
        ss += rmsNormEps;
        ss = (float) (1.0 / Math.sqrt(ss));
        for (int i = 0; i < size; i++) {
            out.setFloat(outOffset + i, ss * x.getFloat(xOffset + i));
        }
    }

    /** Apply Rotary Positional Embeddings (RoPE) to a tensor. */
    static void applyRoPE(FloatTensor tensor, int headSize, int nHeads, int halfHead,
                           FloatBuffer freqsReal, FloatBuffer freqsImag, int ropePos) {
        for (int h = 0; h < nHeads; ++h) {
            int poffset = h * headSize;
            for (int i0 = 0; i0 < halfHead; i0++) {
                float fcr = freqsReal.get(ropePos * halfHead + i0);
                float fci = freqsImag.get(ropePos * halfHead + i0);
                float v0 = tensor.getFloat(poffset + i0);
                float v1 = tensor.getFloat(poffset + i0 + halfHead);
                tensor.setFloat(poffset + i0, v0 * fcr - v1 * fci);
                tensor.setFloat(poffset + i0 + halfHead, v0 * fci + v1 * fcr);
            }
        }
    }

    static void forwardRecurrentLayer(Llama.Configuration config, Llama.Weights weights,
                                      Llama.State state, int layer, int dim) {
        rmsnorm(state.xb, state.x, weights.rms_att_weight[layer], dim, config.rmsNormEps);

        int dConv = config.shortConvLCache;
        int hist = dConv - 1;
        weights.shortConvInProj[layer].matmul(state.xb, state.shortConvTmp, 3 * dim, dim);
        FloatTensor convState = state.shortConvState[layer];
        FloatBuffer kernel = weights.shortConvKernel[layer];

        if (hist == 2) {
            for (int c = 0; c < dim; c++) {
                float b = state.shortConvTmp.getFloat(c);
                float cg = state.shortConvTmp.getFloat(dim + c);
                float xv = state.shortConvTmp.getFloat(2 * dim + c);
                float bx = b * xv;
                int kBase = c * dConv;
                float prev0 = convState.getFloat(c);
                float prev1 = convState.getFloat(dim + c);
                float sum = prev0 * kernel.get(kBase) + prev1 * kernel.get(kBase + 1) + bx * kernel.get(kBase + 2);
                state.xb2.setFloat(c, cg * sum);
                convState.setFloat(c, prev1);
                convState.setFloat(dim + c, bx);
            }
        } else if (hist == 1) {
            for (int c = 0; c < dim; c++) {
                float b = state.shortConvTmp.getFloat(c);
                float cg = state.shortConvTmp.getFloat(dim + c);
                float xv = state.shortConvTmp.getFloat(2 * dim + c);
                float bx = b * xv;
                int kBase = c * dConv;
                float sum = convState.getFloat(c) * kernel.get(kBase) + bx * kernel.get(kBase + 1);
                state.xb2.setFloat(c, cg * sum);
                convState.setFloat(c, bx);
            }
        } else {
            for (int c = 0; c < dim; c++) {
                float b = state.shortConvTmp.getFloat(c);
                float cg = state.shortConvTmp.getFloat(dim + c);
                float xv = state.shortConvTmp.getFloat(2 * dim + c);
                float bx = b * xv;
                int kBase = c * dConv;
                float sum = bx * kernel.get(kBase + dConv - 1);
                for (int k = 0; k < hist; k++) {
                    sum += convState.getFloat(k * dim + c) * kernel.get(kBase + k);
                }
                state.xb2.setFloat(c, cg * sum);
                for (int k = 0; k < hist - 1; k++) {
                    convState.setFloat(k * dim + c, convState.getFloat((k + 1) * dim + c));
                }
                if (hist > 0) {
                    convState.setFloat((hist - 1) * dim + c, bx);
                }
            }
        }

        weights.shortConvOutProj[layer].matmul(state.xb2, state.xb2, dim, dim);
        state.x.addInPlace(state.xb2);
    }

    static void forwardAttentionLayer(Llama.Configuration config, Llama.Weights weights,
                                      Llama.State state, int layer, int position, int dim) {
        boolean layerIsSWA = config.isSWA[layer];
        int headSize = config.headSize(layer);
        int halfHead = headSize / 2;
        int queryDim = config.queryDim(layer);
        int kvDim = config.kvDim(layer);

        rmsnorm(state.xb, state.x, weights.rms_att_weight[layer], dim, config.rmsNormEps);
        weights.wq[layer].matmul(state.xb, state.q, queryDim, dim);
        for (int h = 0; h < config.numberOfHeads; h++) {
            rmsnorm(state.q, h * headSize, state.q, h * headSize, weights.attn_q_norm[layer], headSize, config.rmsNormEps);
        }

        FloatBuffer freqsReal = layerIsSWA ? weights.freq_cis_real_swa : weights.freq_cis_real_full;
        FloatBuffer freqsImag = layerIsSWA ? weights.freq_cis_imag_swa : weights.freq_cis_imag_full;
        int ropePos = Math.max(0, Math.min(config.contextLength - 1, position));
        applyRoPE(state.q, headSize, config.numberOfHeads, halfHead, freqsReal, freqsImag, ropePos);

        int kvLayer = config.kvSourceLayer(layer);
        int nKvHeads = config.numberOfKeyValueHeads(layer);
        int kvMul = config.numberOfHeads / nKvHeads;
        weights.wk[layer].matmul(state.xb, state.k, kvDim, dim);
        if (weights.wv[layer] != null) {
            weights.wv[layer].matmul(state.xb, state.v, kvDim, dim);
        } else {
            state.k.copyTo(0, state.v, 0, kvDim);
        }
        for (int h = 0; h < nKvHeads; h++) {
            rmsnorm(state.k, h * headSize, state.k, h * headSize, weights.attn_k_norm[layer], headSize, config.rmsNormEps);
        }
        applyRoPE(state.k, headSize, nKvHeads, halfHead, freqsReal, freqsImag, ropePos);

        int kvPos = config.kvCacheIndex(layer, position);
        state.k.copyTo(0, state.keyCache[kvLayer], kvPos * kvDim, kvDim);
        state.v.copyTo(0, state.valueCache[kvLayer], kvPos * kvDim, kvDim);

        int attStart = layerIsSWA ? Math.max(0, position - config.slidingWindow + 1) : 0;
        float attnScale = 1.0f / (float) Math.sqrt(headSize);
        Parallel.parallelFor(0, config.numberOfHeads, h -> {
            int qOffset = h * headSize;
            int attOffset = h * config.contextLength;
            int kvHeadOffset = (h / kvMul) * headSize;
            for (int t = attStart; t <= position; t++) {
                int keyCacheOffset = config.kvCacheIndex(layer, t) * kvDim + kvHeadOffset;
                float score = state.q.dot(qOffset, state.keyCache[kvLayer], keyCacheOffset, headSize) * attnScale;
                state.att.setFloat(attOffset + t, score);
            }
            state.att.softmaxInPlace(attOffset + attStart, position - attStart + 1);
            int xbOffset = h * headSize;
            state.xb_k.fillInPlace(xbOffset, headSize, 0f);
            for (int t = attStart; t <= position; t++) {
                int vOffset = config.kvCacheIndex(layer, t) * kvDim + kvHeadOffset;
                float a = state.att.getFloat(attOffset + t);
                state.xb_k.saxpyInPlace(xbOffset, state.valueCache[kvLayer], vOffset, headSize, a);
            }
        });
        weights.wo[layer].matmul(state.xb_k, state.xb2, dim, queryDim);
        if (weights.post_attention_norm[layer] != null) {
            rmsnorm(state.xb2, state.xb2, weights.post_attention_norm[layer], dim, config.rmsNormEps);
        }
        state.x.addInPlace(state.xb2);
    }

    static FloatTensor forward(Llama model, Llama.State state, int token, int position) {
        Llama.Configuration config = model.configuration();
        Llama.Weights weights = model.weights();
        int dim = config.embeddingLength;

        // copy the token embedding into x
        weights.token_embedding_table.copyTo(token * dim, state.x, 0, dim);

        // Compute per-layer inputs (if model has per-layer embeddings)
        int plDim = config.embeddingLengthPerLayer;
        int plTotal = plDim * config.numberOfLayers;
        if (plDim > 0 && weights.perLayerTokenEmbd != null) {
            float sqrtPlDim = (float) Math.sqrt(plDim);
            float projScale = (float) (1.0 / Math.sqrt(dim));
            float inputScale = (float) (1.0 / Math.sqrt(2.0));

            // Project x through perLayerModelProj, scale, and RMS norm per chunk
            weights.perLayerModelProj.matmul(state.x, state.perLayerInputs, plTotal, dim);
            state.perLayerInputs.mapInPlace(0, plTotal, v -> v * projScale);
            for (int l = 0; l < config.numberOfLayers; l++) {
                rmsnorm(state.perLayerInputs, l * plDim, state.perLayerInputs, l * plDim,
                        weights.perLayerProjNorm, plDim, config.rmsNormEps);
            }

            // Add per-layer token embedding scaled by sqrt(plDim)
            long tokEmbOffset = (long) token * plTotal;
            for (int i = 0; i < plTotal; i++) {
                float tokEmb = weights.perLayerTokenEmbd.getFloat(tokEmbOffset + i) * sqrtPlDim;
                state.perLayerInputs.setFloat(i, state.perLayerInputs.getFloat(i) + tokEmb);
            }

            // Scale combined input by 1/sqrt(2)
            state.perLayerInputs.mapInPlace(0, plTotal, v -> v * inputScale);
        }

        // forward all the layers
        for (int l = 0; l < config.numberOfLayers; l++) {
            int hiddenDim = config.feedForwardLength[l];

            if (config.isRecurrentLayer(l)) {
                forwardRecurrentLayer(config, weights, state, l, dim);
            } else {
                forwardAttentionLayer(config, weights, state, l, position, dim);
            }

            // FFN
            rmsnorm(state.xb, state.x, weights.rms_ffn_weight[l], dim, config.rmsNormEps);
            boolean isMoELayer = config.isMoELayer(l);
            if (isMoELayer) {
                // === MoE FFN ===
                // Router: gate_inp @ xb -> sigmoid -> top-k -> normalize weights
                weights.ffnGateInp[l].matmul(state.xb, state.routerLogits, config.expertCount, dim);

                // Add expert probability bias
                if (weights.ffnExpProbsB[l] != null) {
                    for (int i = 0; i < config.expertCount; i++) {
                        state.routerLogits.setFloat(i, state.routerLogits.getFloat(i) + weights.ffnExpProbsB[l].get(i));
                    }
                }

                // Apply gating function (sigmoid for LFM2MOE)
                if (config.expertGatingFunc == 2) {
                    state.routerLogits.mapInPlace(0, config.expertCount, v -> 1.0f / (1.0f + (float) Math.exp(-v)));
                } else {
                    state.routerLogits.softmaxInPlace(0, config.expertCount);
                }

                // Find top-k experts
                int nExperts = config.expertCount;
                int topK = config.expertUsedCount;
                int[] topExperts = state.topExperts;
                float[] topProbs = state.topProbs;
                for (int ki = 0; ki < topK; ki++) {
                    int bestIdx = 0;
                    float bestVal = Float.NEGATIVE_INFINITY;
                    for (int ei = 0; ei < nExperts; ei++) {
                        float val = state.routerLogits.getFloat(ei);
                        if (val > bestVal) {
                            bestVal = val;
                            bestIdx = ei;
                        }
                    }
                    topExperts[ki] = bestIdx;
                    topProbs[ki] = bestVal;
                    state.routerLogits.setFloat(bestIdx, Float.NEGATIVE_INFINITY); // mask for next iteration
                }

                // Normalize weights (norm_w = true for LFM2MOE)
                float weightSum = 0f;
                for (int ki = 0; ki < topK; ki++) weightSum += topProbs[ki];
                for (int ki = 0; ki < topK; ki++) topProbs[ki] /= weightSum;

                // Run selected experts and accumulate
                int expertFF = config.expertFeedForwardLength;
                state.xb2.fillInPlace(0, dim, 0f);

                for (int ki = 0; ki < topK; ki++) {
                    int expertIdx = topExperts[ki];
                    float weight = topProbs[ki];

                    // gate = silu(gate_exps[expert] @ xb)
                    int gateOffset = expertIdx * expertFF * dim;
                    weights.ffnGateExps[l].matmul(state.xb, state.expertGate, expertFF, dim, gateOffset);
                    state.expertGate.mapInPlace(0, expertFF, Llama::silu);

                    // up = up_exps[expert] @ xb
                    int upOffset = expertIdx * expertFF * dim;
                    weights.ffnUpExps[l].matmul(state.xb, state.expertUp, expertFF, dim, upOffset);

                    // gate * up
                    for (int i = 0; i < expertFF; i++) {
                        state.expertGate.setFloat(i, state.expertGate.getFloat(i) * state.expertUp.getFloat(i));
                    }

                    // down = down_exps[expert] @ (gate * up)
                    int downOffset = expertIdx * dim * expertFF;
                    weights.ffnDownExps[l].matmul(state.expertGate, state.expertDown, dim, expertFF, downOffset);

                    // Accumulate: xb2 += weight * expertDown
                    state.xb2.saxpyInPlace(0, state.expertDown, 0, dim, weight);
                }

                // Copy MoE output to xb for residual
                state.xb2.copyTo(0, state.xb, 0, dim);

                // Overall post-FFW norm + residual
                if (weights.post_ffw_norm[l] != null) {
                    rmsnorm(state.xb, state.xb, weights.post_ffw_norm[l], dim, config.rmsNormEps);
                }
                state.x.addInPlace(state.xb);
            } else {
                // Standard dense FFN: w2(SiLU(w1(x)) * w3(x))
                weights.w1[l].matmul(state.xb, state.hb, hiddenDim, dim);
                weights.w3[l].matmul(state.xb, state.hb2, hiddenDim, dim);
                state.hb.mapInPlace(0, hiddenDim, Llama::silu);
                state.hb.multiplyInPlace(0, state.hb2, 0, hiddenDim);
                weights.w2[l].matmul(state.hb, state.xb, dim, hiddenDim);
                if (weights.post_ffw_norm[l] != null) {
                    rmsnorm(state.xb, state.xb, weights.post_ffw_norm[l], dim, config.rmsNormEps);
                }
                state.x.addInPlace(state.xb);
            }

            // Per-layer embedding: GELU-gated projection
            if (plDim > 0 && weights.perLayerInpGate != null) {
                weights.perLayerInpGate[l].matmul(state.x, state.plGate, plDim, dim);
                state.plGate.mapInPlace(0, plDim, Llama::gelu);
                int plOffset = l * plDim;
                for (int i = 0; i < plDim; i++) {
                    state.plGate.setFloat(i, state.plGate.getFloat(i) * state.perLayerInputs.getFloat(plOffset + i));
                }
                weights.perLayerProj[l].matmul(state.plGate, state.plProj, dim, plDim);
                rmsnorm(state.plProj, state.plProj, weights.perLayerPostNorm[l], dim, config.rmsNormEps);
                state.x.addInPlace(state.plProj);
            }

            // Layer output scale
            float scale = weights.layerOutputScale[l];
            if (scale != 1.0f) {
                state.x.mapInPlace(0, dim, v -> v * scale);
            }
        }

        // Final norm + logits
        rmsnorm(state.x, state.x, weights.rms_final_weight, dim, config.rmsNormEps);
        weights.wcls.matmul(state.x, state.logits, config.vocabularySize, dim);
        if (config.logitSoftcapping > 0) {
            float cap = config.logitSoftcapping;
            state.logits.mapInPlace(v -> cap * (float) Math.tanh(v / cap));
        }
        return state.logits;
    }

    private static final String ANSI_CYAN = "\033[36m";
    private static final String ANSI_RESET = "\033[0m";

    public static List<Integer> generateTokens(Llama model, Llama.State state, int startPosition, List<Integer> promptTokens, Set<Integer> stopTokens, int maxTokens, Sampler sampler, boolean echo,
                                               boolean color, IntConsumer onTokenGenerated) {
        long startNanos = System.nanoTime();
        long startGen = promptTokens.isEmpty() ? startNanos : 0;
        if (maxTokens < 0 || model.configuration().contextLength < maxTokens) {
            maxTokens = model.configuration().contextLength;
        }
        List<Integer> generatedTokens = new ArrayList<>(maxTokens);
        int token = state.latestToken; // current token (initialized to BOS)
        int nextToken;
        int promptIndex = 0;
        if (startPosition == 0 && !promptTokens.isEmpty() && promptTokens.getFirst() == token) {
            // avoid feeding BOS twice when prompt explicitly starts with BOS
            promptIndex = 1;
        }
        for (int position = startPosition; position < maxTokens; ++position) {
            forward(model, state, token, position);
            if (promptIndex < promptTokens.size()) {
                nextToken = promptTokens.get(promptIndex++);
                if (echo) {
                    System.err.print(LFMTokenizer.replaceControlCharacters(model.tokenizer().decode(nextToken)));
                }
                if (promptIndex >= promptTokens.size()) {
                    startGen = System.nanoTime();
                }
            } else {
                nextToken = sampler.sampleToken(state.logits);
                if (echo) {
                    System.err.print(LFMTokenizer.replaceControlCharacters(model.tokenizer().decode(nextToken)));
                }
                generatedTokens.add(nextToken);
                if (onTokenGenerated != null) {
                    onTokenGenerated.accept(nextToken);
                }
                if (stopTokens.contains(nextToken)) {
                    break;
                }
            }
            state.latestToken = token = nextToken;
        }

        long elapsedNanos = System.nanoTime() - startNanos;
        long promptNanos = startGen - startNanos;
        long genNanos = elapsedNanos - startGen + startNanos;
        String timingPrefix = color ? ANSI_CYAN : "";
        String timingSuffix = color ? ANSI_RESET : "";
        System.err.printf("%n%scontext: %d/%d prompt: %.2f tokens/s (%d) generation: %.2f tokens/s (%d)%s%n",
                timingPrefix,
                startPosition + promptIndex + generatedTokens.size(), model.configuration().contextLength,
                promptTokens.size() / (promptNanos / 1_000_000_000.0), promptTokens.size(),
                generatedTokens.size() / (genNanos / 1_000_000_000.0), generatedTokens.size(),
                timingSuffix);

        return generatedTokens;
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

final class Parallel {
    public static void parallelFor(int startInclusive, int endExclusive, IntConsumer action) {
        IntStream.range(startInclusive, endExclusive).parallel().forEach(action);
    }
}

record Pair<First, Second>(First first, Second second) {
}

record GGMLTensorEntry(MemorySegment mappedFile, String name, GGMLType ggmlType, int[] shape,
                       MemorySegment memorySegment) {
}

final class Float16 {
    public static final int BYTES = 2;
}

enum GGMLType {
    F32(Float.BYTES),          // 0
    F16(Float16.BYTES),        // 1
    Q4_0(Float16.BYTES + 16 * Byte.BYTES, 32),  // 2
    Q4_1(2 * Float16.BYTES + 16 * Byte.BYTES, 32), // 3
    UNSUPPORTED_Q4_2(Integer.MAX_VALUE), // 4 - removed
    UNSUPPORTED_Q4_3(Integer.MAX_VALUE), // 5 - removed
    Q5_0(Integer.MAX_VALUE),   // 6
    Q5_1(2 * Float16.BYTES + Integer.BYTES + 16 * Byte.BYTES, 32),   // 7
    Q8_0(Float16.BYTES + 32 * Byte.BYTES, 32),  // 8
    Q8_1(32 * Byte.BYTES + 2 * Float.BYTES, 32), // 9
    Q2_K(Integer.MAX_VALUE),   // 10
    Q3_K(Integer.MAX_VALUE),   // 11
    Q4_K(2 * Float16.BYTES + ((GGMLType.QK_K / 16) / 8 * 6) + GGMLType.QK_K / 2, GGMLType.QK_K), // 12
    Q5_K(2 * Float16.BYTES + ((GGMLType.QK_K / 16) / 8 * 6) + GGMLType.QK_K / 8 + GGMLType.QK_K / 2, GGMLType.QK_K), // 13
    Q6_K(GGMLType.QK_K / 2 + GGMLType.QK_K / 4 + GGMLType.QK_K / 16 + Float16.BYTES, GGMLType.QK_K), // 14
    Q8_K(Integer.MAX_VALUE),   // 15
    IQ2_XXS(Integer.MAX_VALUE), // 16
    IQ2_XS(Integer.MAX_VALUE),  // 17
    IQ3_XXS(Integer.MAX_VALUE), // 18
    IQ1_S(Integer.MAX_VALUE),   // 19
    IQ4_NL(Integer.MAX_VALUE),  // 20
    IQ3_S(Integer.MAX_VALUE),   // 21
    IQ2_S(Integer.MAX_VALUE),   // 22
    IQ4_XS(Integer.MAX_VALUE),  // 23
    I8(Byte.BYTES),             // 24
    I16(Short.BYTES),           // 25
    I32(Integer.BYTES),         // 26
    I64(Long.BYTES),            // 27
    F64(Double.BYTES),          // 28
    IQ1_M(Integer.MAX_VALUE),   // 29
    BF16(Float16.BYTES),        // 30
    UNSUPPORTED_Q4_0_4_4(Integer.MAX_VALUE), // 31 - removed from gguf files
    UNSUPPORTED_Q4_0_4_8(Integer.MAX_VALUE), // 32
    UNSUPPORTED_Q4_0_8_8(Integer.MAX_VALUE), // 33
    TQ1_0(Integer.MAX_VALUE),   // 34
    TQ2_0(Integer.MAX_VALUE),   // 35
    UNSUPPORTED_IQ4_NL_4_4(Integer.MAX_VALUE), // 36
    UNSUPPORTED_IQ4_NL_4_8(Integer.MAX_VALUE), // 37
    UNSUPPORTED_IQ4_NL_8_8(Integer.MAX_VALUE), // 38
    MXFP4(Byte.BYTES + GGMLType.QK_MXFP4 / 2, GGMLType.QK_MXFP4), // 39
    NVFP4(Integer.MAX_VALUE);   // 40

    private static final GGMLType[] VALUES = values();

    private final int typeSize;

    private final int blockSize;

    public int getTypeSize() {
        return typeSize;
    }

    public int getBlockSize() {
        return blockSize;
    }

    public static GGMLType fromId(int id) {
        if (0 <= id && id < VALUES.length) {
            return VALUES[id];
        }
        throw new UnsupportedOperationException("Unsupported GGML tensor type id: " + id);
    }

    GGMLType(int typeSize) {
        this(typeSize, 1);
    }

    public long byteSizeFor(long numberOfElements) {
        long t = numberOfElements * (long) getTypeSize();
        assert t % getBlockSize() == 0;
        return t / getBlockSize();
    }

    public static final int QK_K = 256;
    public static final int QK_MXFP4 = 32;

    GGMLType(int typeSize, int blockSize) {
        assert blockSize > 0;
        assert typeSize > 0;
        assert isPowerOf2(blockSize);
        this.typeSize = typeSize;
        this.blockSize = blockSize;
    }

    private static boolean isPowerOf2(int n) {
        return n > 0 && (n & (n - 1)) == 0;
    }
}

abstract class FloatTensor {
    static final int VECTOR_BIT_SIZE = Integer.getInteger("llama.VectorBitSize", VectorShape.preferredShape().vectorBitSize());
    static final boolean USE_VECTOR_API = VECTOR_BIT_SIZE != 0;

    static final VectorSpecies<Float> F_SPECIES;
    static final VectorSpecies<Integer> I_SPECIES;
    static final VectorSpecies<Short> S_SPECIES_HALF;

    static {
        if (USE_VECTOR_API) {
            F_SPECIES = VectorShape.forBitSize(VECTOR_BIT_SIZE).withLanes(float.class);
            I_SPECIES = F_SPECIES.withLanes(int.class);
            S_SPECIES_HALF = VectorShape.forBitSize(F_SPECIES.vectorBitSize() / 2).withLanes(short.class);
            assert F_SPECIES.length() == S_SPECIES_HALF.length();
        } else {
            F_SPECIES = null;
            I_SPECIES = null;
            S_SPECIES_HALF = null;
        }
    }

    static final sun.misc.Unsafe UNSAFE;

    static {
        try {
            Field f = sun.misc.Unsafe.class.getDeclaredField("theUnsafe");
            f.setAccessible(true);
            UNSAFE = (sun.misc.Unsafe) f.get(null);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            throw new RuntimeException(e);
        }
    }

    static short readShort(MemorySegment memorySegment, long offset) {
        return UNSAFE.getShort(memorySegment.address() + offset);
    }

    static void writeShort(MemorySegment memorySegment, long offset, short value) {
        UNSAFE.putShort(memorySegment.address() + offset, value);
    }

    static float readFloat16(MemorySegment memorySegment, long offset) {
        return Float.float16ToFloat(readShort(memorySegment, offset));
    }

    static byte readByte(MemorySegment memorySegment, long offset) {
        return UNSAFE.getByte(memorySegment.address() + offset);
    }

    static float readFloat(MemorySegment memorySegment, long offset) {
        return UNSAFE.getFloat(memorySegment.address() + offset);
    }

    abstract long size();

    abstract float getFloat(long index);

    abstract void setFloat(int index, float value);

    abstract FloatVector getFloatVector(VectorSpecies<Float> species, int offset);

    abstract GGMLType type();

    public static int numberOfElements(int... dimensions) {
        assert Arrays.stream(dimensions).allMatch(i -> i > 0);
        return Arrays.stream(dimensions).reduce(Math::multiplyExact).orElseThrow();
    }

    public static long numberOfElementsLong(int... dimensions) {
        long result = 1;
        for (int d : dimensions) {
            assert d > 0;
            result = Math.multiplyExact(result, d);
        }
        return result;
    }

    static float scalarDot(FloatTensor thiz, int thisOffset, FloatTensor that, int thatOffset, int size) {
        float result = 0f;
        for (int j = 0; j < size; j++) {
            result += thiz.getFloat(thisOffset + j) * that.getFloat(thatOffset + j);
        }
        return result;
    }

    float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        return scalarDot(this, thisOffset, that, thatOffset, size);
    }

    void matmul(FloatTensor that, FloatTensor out, int dim0, int dim1) {
        matmul(that, out, dim0, dim1, 0);
    }

    // matmul with offset into this tensor (for expert weight slicing in 3D tensors)
    void matmul(FloatTensor that, FloatTensor out, int dim0, int dim1, int thisOffset) {
        if (that == out) {
            // In-place matmul: must avoid read-after-write races in parallel execution
            float[] temp = new float[dim0];
            Parallel.parallelFor(0, dim0, i -> temp[i] = dot(thisOffset + i * dim1, that, 0, dim1));
            for (int i = 0; i < dim0; i++) {
                out.setFloat(i, temp[i]);
            }
        } else {
            Parallel.parallelFor(0, dim0, i -> out.setFloat(i, dot(thisOffset + i * dim1, that, 0, dim1)));
        }
    }

    @FunctionalInterface
    interface AggregateFunction {
        float apply(float acc, float value);
    }

    float reduce(int thisOffset, int size, float seed, AggregateFunction reduce) {
        float result = seed;
        for (int i = 0; i < size; ++i) {
            result = reduce.apply(result, getFloat(thisOffset + i));
        }
        return result;
    }

    float sum(int thisOffset, int size) {
        return reduce(thisOffset, size, 0f, Float::sum);
    }

    float max(int thisOffset, int size) {
        return reduce(thisOffset, size, Float.NEGATIVE_INFINITY, Float::max);
    }

    void copyTo(int thisOffset, FloatTensor that, int thatOffset, int size) {
        that.mapWithIndexInPlace(thatOffset, size, (value, index) -> this.getFloat(index - thatOffset + thisOffset));
    }

    int argmax(int thisOffset, int size) {
        assert size > 0;
        int maxIndex = thisOffset;
        float maxValue = this.getFloat(maxIndex);
        int endIndex = thisOffset + size;
        for (int i = thisOffset; i < endIndex; ++i) {
            float f = this.getFloat(i);
            if (f > maxValue) {
                maxValue = f;
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    int argmax() {
        return argmax(0, Math.toIntExact(size()));
    }

    @FunctionalInterface
    interface MapFunction {
        float apply(float value);
    }

    @FunctionalInterface
    interface MapWithIndexFunction {
        float apply(float value, int index);
    }

    FloatTensor mapInPlace(int thisOffset, int size, MapFunction mapFunction) {
        int endIndex = thisOffset + size;
        for (int i = thisOffset; i < endIndex; ++i) {
            setFloat(i, mapFunction.apply(getFloat(i)));
        }
        return this;
    }

    FloatTensor mapInPlace(MapFunction mapFunction) {
        return mapInPlace(0, Math.toIntExact(size()), mapFunction);
    }

    FloatTensor mapWithIndexInPlace(int thisOffset, int size, FloatTensor.MapWithIndexFunction mapWithIndexFunction) {
        int endOffset = thisOffset + size;
        for (int i = thisOffset; i < endOffset; ++i) {
            setFloat(i, mapWithIndexFunction.apply(getFloat(i), i));
        }
        return this;
    }

    FloatTensor addInPlace(int thisOffset, FloatTensor that, int thatOffset, int size) {
        return mapWithIndexInPlace(thisOffset, size, (value, index) -> value + that.getFloat(index - thisOffset + thatOffset));
    }

    FloatTensor addInPlace(FloatTensor that) {
        return addInPlace(0, that, 0, Math.toIntExact(size()));
    }

    FloatTensor multiplyInPlace(int thisOffset, FloatTensor that, int thatOffset, int size) {
        return mapWithIndexInPlace(thisOffset, size, (value, index) -> value * that.getFloat(index - thisOffset + thatOffset));
    }

    FloatTensor divideInPlace(int thisOffset, int size, float value) {
        return mapInPlace(thisOffset, size, f -> f / value);
    }

    FloatTensor fillInPlace(int thisOffset, int size, float value) {
        return mapInPlace(thisOffset, size, unused -> value);
    }

    FloatTensor softmaxInPlace(int thisOffset, int size) {
        float maxVal = max(thisOffset, size);
        mapInPlace(thisOffset, size, f -> (float) Math.exp(f - maxVal));
        float sum = sum(thisOffset, size);
        return divideInPlace(thisOffset, size, sum);
    }

    FloatTensor saxpyInPlace(int thisOffset, FloatTensor that, int thatOffset, int size, float a) {
        for (int i = 0; i < size; ++i) {
            setFloat(thisOffset + i, a * that.getFloat(thatOffset + i) + this.getFloat(thisOffset + i));
        }
        return this;
    }
}

final class Q4_0FloatTensor extends FloatTensor {

    final long size;
    final MemorySegment memorySegment;

    public Q4_0FloatTensor(long size, MemorySegment memorySegment) {
        this.size = size;
        this.memorySegment = memorySegment;
    }

    @Override
    long size() {
        return size;
    }

    @Override
    public void setFloat(int index, float value) {
        throw new UnsupportedOperationException("setFloat");
    }

    @Override
    FloatVector getFloatVector(VectorSpecies<Float> species, int index) {
        throw new UnsupportedOperationException("getFloatVector");
    }

    @Override
    public GGMLType type() {
        return GGMLType.Q4_0;
    }

    @Override
    public float getFloat(long index) {
        assert 0 <= index && index < size;
        long blockIndex = index / GGMLType.Q4_0.getBlockSize();
        long blockOffset = blockIndex * GGMLType.Q4_0.getTypeSize();
        float scale = readFloat16(memorySegment, blockOffset);
        byte quant;
        int modIndex = (int) (index % GGMLType.Q4_0.getBlockSize());
        if (modIndex < GGMLType.Q4_0.getBlockSize() / 2) {
            quant = (byte) (readByte(memorySegment, blockOffset + Float16.BYTES + modIndex) & 0x0F);
        } else {
            quant = (byte) ((readByte(memorySegment, blockOffset + Float16.BYTES + modIndex - GGMLType.Q4_0.getBlockSize() / 2) >>> 4) & 0x0F);
        }
        quant -= 8;
        return quant * scale;
    }

    @Override
    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        if (FloatTensor.USE_VECTOR_API) {
            return vectorDot(this, thisOffset, (ArrayFloatTensor) that, thatOffset, size);
        } else {
            return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
        }
    }

    private static float vectorDot(Q4_0FloatTensor thiz, int thisOffset, ArrayFloatTensor that, int thatOffset, int size) {
        float result = 0f;
        int j = 0;

        assert Integer.bitCount(GGMLType.Q4_0.getBlockSize()) == 1 : "power of 2";
        int alignmentBound = Math.min(size, -thisOffset & (GGMLType.Q4_0.getBlockSize() - 1));
        if (alignmentBound > 0) {
            result += FloatTensor.scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound);
            j += alignmentBound;
        }
        assert (thisOffset + j) % GGMLType.Q4_0.getBlockSize() == 0;

        FloatVector val = FloatVector.zero(F_SPECIES);
        long blockOffset = (long) (thisOffset + j) / GGMLType.Q4_0.getBlockSize() * GGMLType.Q4_0.getTypeSize();
        int upperBound = j + (size - j) / GGMLType.Q4_0.getBlockSize() * GGMLType.Q4_0.getBlockSize();
        for (; j < upperBound; j += GGMLType.Q4_0.getBlockSize(), blockOffset += GGMLType.Q4_0.getTypeSize()) {
            float wScaleValue = readFloat16(thiz.memorySegment, blockOffset);
            var wScale = FloatVector.broadcast(F_SPECIES, wScaleValue);
            var wBytes = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, thiz.memorySegment, blockOffset + Float16.BYTES, ByteOrder.LITTLE_ENDIAN);
            var loBytes = wBytes.and((byte) 0xF).sub((byte) 8);
            var hiBytes = wBytes.lanewise(VectorOperators.LSHR, 4).sub((byte) 8);

            switch (F_SPECIES.vectorBitSize()) {
                case 512 -> {
                    var s0 = that.getFloatVector(F_SPECIES, thatOffset + j).mul(loBytes.castShape(F_SPECIES, 0));
                    var s1 = that.getFloatVector(F_SPECIES, thatOffset + j + F_SPECIES.length()).mul(hiBytes.castShape(F_SPECIES, 0));
                    val = s0.add(s1).fma(wScale, val);
                }
                case 256 -> {
                    var s0 = that.getFloatVector(F_SPECIES, thatOffset + j).mul(loBytes.castShape(F_SPECIES, 0));
                    var s1 = that.getFloatVector(F_SPECIES, thatOffset + j + 2 * F_SPECIES.length()).mul(hiBytes.castShape(F_SPECIES, 0));
                    s0 = that.getFloatVector(F_SPECIES, thatOffset + j + F_SPECIES.length()).fma(loBytes.castShape(F_SPECIES, 1), s0);
                    s1 = that.getFloatVector(F_SPECIES, thatOffset + j + 3 * F_SPECIES.length()).fma(hiBytes.castShape(F_SPECIES, 1), s1);
                    val = s0.add(s1).fma(wScale, val);
                }
                case 128 -> {
                    for (int i = 0; i < 2; ++i) {
                        var tmp = i == 0 ? loBytes : hiBytes;
                        var s0 = that.getFloatVector(F_SPECIES, thatOffset + j + (i * 4) * F_SPECIES.length()).mul(tmp.castShape(F_SPECIES, 0));
                        var s1 = that.getFloatVector(F_SPECIES, thatOffset + j + (i * 4 + 2) * F_SPECIES.length()).mul(tmp.castShape(F_SPECIES, 2));
                        s0 = that.getFloatVector(F_SPECIES, thatOffset + j + (i * 4 + 1) * F_SPECIES.length()).fma(tmp.castShape(F_SPECIES, 1), s0);
                        s1 = that.getFloatVector(F_SPECIES, thatOffset + j + (i * 4 + 3) * F_SPECIES.length()).fma(tmp.castShape(F_SPECIES, 3), s1);
                        val = s0.add(s1).fma(wScale, val);
                    }
                }
                default -> throw new UnsupportedOperationException(F_SPECIES.toString());
            }
        }
        result += val.reduceLanes(VectorOperators.ADD);

        if (j < size) {
            result += FloatTensor.scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j);
        }

        return result;
    }
}

final class Q4_1FloatTensor extends FloatTensor {

    final long size;
    final MemorySegment memorySegment;

    public Q4_1FloatTensor(long size, MemorySegment memorySegment) {
        this.size = size;
        this.memorySegment = memorySegment;
    }

    @Override long size() { return size; }
    @Override public void setFloat(int index, float value) { throw new UnsupportedOperationException("setFloat"); }
    @Override FloatVector getFloatVector(VectorSpecies<Float> species, int index) { throw new UnsupportedOperationException("getFloatVector"); }
    @Override public GGMLType type() { return GGMLType.Q4_1; }

    @Override
    public float getFloat(long index) {
        assert 0 <= index && index < size;
        long blockIndex = index / GGMLType.Q4_1.getBlockSize();
        long blockOffset = blockIndex * GGMLType.Q4_1.getTypeSize();
        float delta = readFloat16(memorySegment, blockOffset);
        float min = readFloat16(memorySegment, blockOffset + Float16.BYTES);
        int modIndex = (int) (index % GGMLType.Q4_1.getBlockSize());
        int quant;
        if (modIndex < 16) {
            quant = Byte.toUnsignedInt(readByte(memorySegment, blockOffset + 2 * Float16.BYTES + modIndex)) & 0x0F;
        } else {
            quant = (Byte.toUnsignedInt(readByte(memorySegment, blockOffset + 2 * Float16.BYTES + modIndex - 16)) >>> 4) & 0x0F;
        }
        return delta * quant + min;
    }

    @Override
    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        if (FloatTensor.USE_VECTOR_API) {
            return vectorDot(this, thisOffset, (ArrayFloatTensor) that, thatOffset, size);
        } else {
            return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
        }
    }

    private static float vectorDot(Q4_1FloatTensor thiz, int thisOffset, ArrayFloatTensor that, int thatOffset, int size) {
        float result = 0f;
        int j = 0;

        assert Integer.bitCount(GGMLType.Q4_1.getBlockSize()) == 1 : "power of 2";
        int alignmentBound = Math.min(size, -thisOffset & (GGMLType.Q4_1.getBlockSize() - 1));
        if (alignmentBound > 0) {
            result += FloatTensor.scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound);
            j += alignmentBound;
        }
        assert (thisOffset + j) % GGMLType.Q4_1.getBlockSize() == 0;

        FloatVector val = FloatVector.zero(F_SPECIES);
        long blockOffset = (long) (thisOffset + j) / GGMLType.Q4_1.getBlockSize() * GGMLType.Q4_1.getTypeSize();
        int upperBound = j + (size - j) / GGMLType.Q4_1.getBlockSize() * GGMLType.Q4_1.getBlockSize();
        for (; j < upperBound; j += GGMLType.Q4_1.getBlockSize(), blockOffset += GGMLType.Q4_1.getTypeSize()) {
            float deltaValue = readFloat16(thiz.memorySegment, blockOffset);
            float minValue = readFloat16(thiz.memorySegment, blockOffset + Float16.BYTES);
            var wDelta = FloatVector.broadcast(F_SPECIES, deltaValue);
            var wMin = FloatVector.broadcast(F_SPECIES, minValue);
            var wBytes = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, thiz.memorySegment, blockOffset + 2 * Float16.BYTES, ByteOrder.LITTLE_ENDIAN);
            var loBytes = wBytes.and((byte) 0xF);
            var hiBytes = wBytes.lanewise(VectorOperators.LSHR, 4);
            switch (F_SPECIES.vectorBitSize()) {
                case 512 -> {
                    var that0 = that.getFloatVector(F_SPECIES, thatOffset + j);
                    var that1 = that.getFloatVector(F_SPECIES, thatOffset + j + F_SPECIES.length());
                    var s0 = that0.mul(loBytes.castShape(F_SPECIES, 0));
                    var s1 = that1.mul(hiBytes.castShape(F_SPECIES, 0));
                    val = s0.add(s1).fma(wDelta, val);
                    val = that0.add(that1).fma(wMin, val);
                }
                case 256 -> {
                    var that0 = that.getFloatVector(F_SPECIES, thatOffset + j);
                    var that1 = that.getFloatVector(F_SPECIES, thatOffset + j + F_SPECIES.length());
                    var that2 = that.getFloatVector(F_SPECIES, thatOffset + j + 2 * F_SPECIES.length());
                    var that3 = that.getFloatVector(F_SPECIES, thatOffset + j + 3 * F_SPECIES.length());
                    var s0 = that0.mul(loBytes.castShape(F_SPECIES, 0));
                    var s1 = that2.mul(hiBytes.castShape(F_SPECIES, 0));
                    s0 = that1.fma(loBytes.castShape(F_SPECIES, 1), s0);
                    s1 = that3.fma(hiBytes.castShape(F_SPECIES, 1), s1);
                    val = s0.add(s1).fma(wDelta, val);
                    val = that0.add(that1).add(that2).add(that3).fma(wMin, val);
                }
                case 128 -> {
                    for (int i = 0; i < 2; ++i) {
                        var tmp = i == 0 ? loBytes : hiBytes;
                        var s0 = that.getFloatVector(F_SPECIES, thatOffset + j + (i * 4) * F_SPECIES.length()).mul(tmp.castShape(F_SPECIES, 0));
                        var s1 = that.getFloatVector(F_SPECIES, thatOffset + j + (i * 4 + 2) * F_SPECIES.length()).mul(tmp.castShape(F_SPECIES, 2));
                        s0 = that.getFloatVector(F_SPECIES, thatOffset + j + (i * 4 + 1) * F_SPECIES.length()).fma(tmp.castShape(F_SPECIES, 1), s0);
                        s1 = that.getFloatVector(F_SPECIES, thatOffset + j + (i * 4 + 3) * F_SPECIES.length()).fma(tmp.castShape(F_SPECIES, 3), s1);
                        val = s0.add(s1).fma(wDelta, val);
                    }
                    // vectorized min contribution
                    var thatSum = FloatVector.zero(F_SPECIES);
                    for (int k = 0; k < GGMLType.Q4_1.getBlockSize(); k += F_SPECIES.length()) {
                        thatSum = thatSum.add(that.getFloatVector(F_SPECIES, thatOffset + j + k));
                    }
                    val = thatSum.fma(wMin, val);
                }
                default -> throw new UnsupportedOperationException(F_SPECIES.toString());
            }
        }
        result += val.reduceLanes(VectorOperators.ADD);

        if (j < size) {
            result += FloatTensor.scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j);
        }

        return result;
    }
}

final class Q5_1FloatTensor extends FloatTensor {

    private final long size;
    private final MemorySegment memorySegment;

    Q5_1FloatTensor(long size, MemorySegment memorySegment) {
        this.size = size;
        this.memorySegment = memorySegment;
    }

    @Override long size() { return size; }
    @Override public void setFloat(int index, float value) { throw new UnsupportedOperationException("setFloat"); }
    @Override FloatVector getFloatVector(VectorSpecies<Float> species, int index) { throw new UnsupportedOperationException("getFloatVector"); }
    @Override public GGMLType type() { return GGMLType.Q5_1; }

    @Override
    public float getFloat(long index) {
        assert 0 <= index && index < size;
        long blockIndex = index / GGMLType.Q5_1.getBlockSize();
        int inBlockIndex = (int) (index % GGMLType.Q5_1.getBlockSize());
        long blockOffset = blockIndex * GGMLType.Q5_1.getTypeSize();

        float d = readFloat16(memorySegment, blockOffset);
        float m = readFloat16(memorySegment, blockOffset + Float16.BYTES);
        int qh = readInt32LE(memorySegment, blockOffset + 2L * Float16.BYTES);

        int j;
        int nibble;
        int xh;
        if (inBlockIndex < GGMLType.Q5_1.getBlockSize() / 2) {
            j = inBlockIndex;
            nibble = Byte.toUnsignedInt(readByte(memorySegment, blockOffset + 2L * Float16.BYTES + Integer.BYTES + j)) & 0x0F;
            xh = ((qh >> j) << 4) & 0x10;
        } else {
            j = inBlockIndex - GGMLType.Q5_1.getBlockSize() / 2;
            nibble = (Byte.toUnsignedInt(readByte(memorySegment, blockOffset + 2L * Float16.BYTES + Integer.BYTES + j)) >>> 4) & 0x0F;
            xh = (qh >> (j + 12)) & 0x10;
        }

        int q = nibble | xh;
        return q * d + m;
    }

    @Override
    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        if (that instanceof ArrayFloatTensor aft) {
            if (FloatTensor.USE_VECTOR_API) {
                return vectorDot(this, thisOffset, aft, thatOffset, size);
            }
            return scalarDot(this, thisOffset, aft, thatOffset, size);
        }
        return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
    }

    private static float vectorDot(Q5_1FloatTensor thiz, int thisOffset, ArrayFloatTensor that, int thatOffset, int size) {
        assert Integer.bitCount(GGMLType.Q5_1.getBlockSize()) == 1 : "power of 2";
        int j = 0;
        float result = 0f;

        int alignmentBound = Math.min(size, -thisOffset & (GGMLType.Q5_1.getBlockSize() - 1));
        if (alignmentBound > 0) {
            result += scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound);
            j = alignmentBound;
        }

        float[] decoded = new float[GGMLType.Q5_1.getBlockSize()];
        int upperBound = j + (size - j) / GGMLType.Q5_1.getBlockSize() * GGMLType.Q5_1.getBlockSize();
        int vecUpper = F_SPECIES.loopBound(GGMLType.Q5_1.getBlockSize());
        for (; j < upperBound; j += GGMLType.Q5_1.getBlockSize()) {
            assert (thisOffset + j) % GGMLType.Q5_1.getBlockSize() == 0;
            long blockOffset = (long) (thisOffset + j) / GGMLType.Q5_1.getBlockSize() * GGMLType.Q5_1.getTypeSize();
            float d = readFloat16(thiz.memorySegment, blockOffset);
            float m = readFloat16(thiz.memorySegment, blockOffset + Float16.BYTES);
            int qh = readInt32LE(thiz.memorySegment, blockOffset + 2L * Float16.BYTES);
            long qsBase = blockOffset + 2L * Float16.BYTES + Integer.BYTES;

            for (int p = 0; p < GGMLType.Q5_1.getBlockSize() / 2; p++) {
                int packed = Byte.toUnsignedInt(readByte(thiz.memorySegment, qsBase + p));
                int x0 = (packed & 0x0F) | ((((qh >> p) << 4) & 0x10));
                int x1 = ((packed >>> 4) & 0x0F) | ((qh >> (p + 12)) & 0x10);
                decoded[p] = x0 * d + m;
                decoded[p + GGMLType.Q5_1.getBlockSize() / 2] = x1 * d + m;
            }

            FloatVector acc = FloatVector.zero(F_SPECIES);
            for (int i = 0; i < vecUpper; i += F_SPECIES.length()) {
                FloatVector w = FloatVector.fromArray(F_SPECIES, decoded, i);
                FloatVector x = that.getFloatVector(F_SPECIES, thatOffset + j + i);
                acc = w.fma(x, acc);
            }
            result += acc.reduceLanes(VectorOperators.ADD);

            for (int i = vecUpper; i < GGMLType.Q5_1.getBlockSize(); i++) {
                result += decoded[i] * that.values[thatOffset + j + i];
            }
        }

        if (j < size) {
            result += scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j);
        }
        return result;
    }

    private static float scalarDot(Q5_1FloatTensor thiz, int thisOffset, ArrayFloatTensor that, int thatOffset, int size) {
        float result = 0f;
        for (int i = 0; i < size; i++) {
            result += thiz.getFloat(thisOffset + i) * that.values[thatOffset + i];
        }
        return result;
    }

    private static int readInt32LE(MemorySegment memorySegment, long offset) {
        int b0 = Byte.toUnsignedInt(readByte(memorySegment, offset));
        int b1 = Byte.toUnsignedInt(readByte(memorySegment, offset + 1));
        int b2 = Byte.toUnsignedInt(readByte(memorySegment, offset + 2));
        int b3 = Byte.toUnsignedInt(readByte(memorySegment, offset + 3));
        return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
    }
}

final class Q4_KFloatTensor extends FloatTensor {

    static final int BLOCK_SIZE = GGMLType.QK_K;
    static final int TYPE_SIZE = GGMLType.Q4_K.getTypeSize();

    final long size;
    final MemorySegment memorySegment;

    public Q4_KFloatTensor(long size, MemorySegment memorySegment) {
        this.size = size;
        this.memorySegment = memorySegment;
    }

    @Override long size() { return size; }
    @Override public void setFloat(int index, float value) { throw new UnsupportedOperationException("setFloat"); }
    @Override FloatVector getFloatVector(VectorSpecies<Float> species, int index) { throw new UnsupportedOperationException("getFloatVector"); }
    @Override public GGMLType type() { return GGMLType.Q4_K; }

    // Decode scale or min for sub-block j (0..7) from the 12-byte scales array
    static int getScaleMinK4(int j, MemorySegment mem, long scalesOffset, boolean isMin) {
        if (j < 4) {
            int idx = isMin ? j + 4 : j;
            return Byte.toUnsignedInt(readByte(mem, scalesOffset + idx)) & 63;
        } else {
            int lowIdx = j + 4;
            int highIdx = isMin ? j : j - 4;
            int low = isMin
                    ? (Byte.toUnsignedInt(readByte(mem, scalesOffset + lowIdx)) >> 4)
                    : (Byte.toUnsignedInt(readByte(mem, scalesOffset + lowIdx)) & 0xF);
            int high = (Byte.toUnsignedInt(readByte(mem, scalesOffset + highIdx)) >> 6) & 0x3;
            return low | (high << 4);
        }
    }

    @Override
    public float getFloat(long index) {
        long blockIndex = index / BLOCK_SIZE;
        int withinBlock = (int) (index % BLOCK_SIZE);
        long blockOffset = blockIndex * TYPE_SIZE;
        float d = readFloat16(memorySegment, blockOffset);
        float dmin = readFloat16(memorySegment, blockOffset + 2);
        long scalesOffset = blockOffset + 4;
        long qsOffset = blockOffset + 16; // 4 + 12

        // Each group of 64 values uses 2 sub-blocks: low nibble (32) + high nibble (32)
        int group = withinBlock / 64;   // 0..3
        int inGroup = withinBlock % 64;
        int subBlock;
        int nibbleIndex;
        boolean isHigh;
        if (inGroup < 32) {
            subBlock = group * 2;
            nibbleIndex = inGroup;
            isHigh = false;
        } else {
            subBlock = group * 2 + 1;
            nibbleIndex = inGroup - 32;
            isHigh = true;
        }

        int sc = getScaleMinK4(subBlock, memorySegment, scalesOffset, false);
        int m = getScaleMinK4(subBlock, memorySegment, scalesOffset, true);

        byte qsByte = readByte(memorySegment, qsOffset + group * 32 + nibbleIndex);
        int quant = isHigh ? ((Byte.toUnsignedInt(qsByte) >> 4) & 0xF) : (Byte.toUnsignedInt(qsByte) & 0xF);

        return d * sc * quant - dmin * m;
    }

    @Override
    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        if (FloatTensor.USE_VECTOR_API) {
            return vectorDot(this, thisOffset, (ArrayFloatTensor) that, thatOffset, size);
        } else {
            return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
        }
    }

    private static float vectorDot(Q4_KFloatTensor thiz, int thisOffset, ArrayFloatTensor that, int thatOffset, int size) {
        float result = 0f;
        int j = 0;

        // Handle unaligned head
        assert Integer.bitCount(BLOCK_SIZE) == 1 : "power of 2";
        int alignmentBound = Math.min(size, -thisOffset & (BLOCK_SIZE - 1));
        if (alignmentBound > 0) {
            result += FloatTensor.scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound);
            j += alignmentBound;
        }

        FloatVector val = FloatVector.zero(F_SPECIES);
        FloatVector val2 = FloatVector.zero(F_SPECIES);
        long blockOffset = (long) (thisOffset + j) / BLOCK_SIZE * TYPE_SIZE;
        int upperBound = j + (size - j) / BLOCK_SIZE * BLOCK_SIZE;

        for (; j < upperBound; j += BLOCK_SIZE, blockOffset += TYPE_SIZE) {
            float d = readFloat16(thiz.memorySegment, blockOffset);
            float dmin = readFloat16(thiz.memorySegment, blockOffset + 2);
            long scalesOff = blockOffset + 4;
            long qsOff = blockOffset + 16;

            // 4 groups of 64 values each (2 sub-blocks per group: low nibble + high nibble)
            for (int g = 0; g < 4; g++) {
                float d1 = d * getScaleMinK4(g * 2, thiz.memorySegment, scalesOff, false);
                float negM1 = -(dmin * getScaleMinK4(g * 2, thiz.memorySegment, scalesOff, true));
                float d2 = d * getScaleMinK4(g * 2 + 1, thiz.memorySegment, scalesOff, false);
                float negM2 = -(dmin * getScaleMinK4(g * 2 + 1, thiz.memorySegment, scalesOff, true));

                var d1Vec = FloatVector.broadcast(F_SPECIES, d1);
                var negM1Vec = FloatVector.broadcast(F_SPECIES, negM1);
                var d2Vec = FloatVector.broadcast(F_SPECIES, d2);
                var negM2Vec = FloatVector.broadcast(F_SPECIES, negM2);

                int loBase = thatOffset + j + g * 64;
                int hiBase = thatOffset + j + g * 64 + 32;

                // Process 32 bytes of qs in 2 chunks of 16 bytes
                for (int c = 0; c < 2; c++) {
                    var wBytes = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, thiz.memorySegment,
                            qsOff + (long) g * 32 + c * 16, ByteOrder.LITTLE_ENDIAN);
                    var loBytes = wBytes.and((byte) 0xF);
                    var hiBytes = wBytes.lanewise(VectorOperators.LSHR, 4);

                    int loIdx = loBase + c * 16;
                    int hiIdx = hiBase + c * 16;

                    switch (F_SPECIES.vectorBitSize()) {
                        case 512 -> {
                            var loQ = loBytes.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            val = loQ.fma(d1Vec, negM1Vec).fma(that.getFloatVector(F_SPECIES, loIdx), val);
                            var hiQ = hiBytes.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            val2 = hiQ.fma(d2Vec, negM2Vec).fma(that.getFloatVector(F_SPECIES, hiIdx), val2);
                        }
                        case 256 -> {
                            var loQ0 = loBytes.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            var loQ1 = loBytes.castShape(F_SPECIES, 1).reinterpretAsFloats();
                            val = loQ0.fma(d1Vec, negM1Vec).fma(that.getFloatVector(F_SPECIES, loIdx), val);
                            val2 = loQ1.fma(d1Vec, negM1Vec).fma(that.getFloatVector(F_SPECIES, loIdx + F_SPECIES.length()), val2);
                            var hiQ0 = hiBytes.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            var hiQ1 = hiBytes.castShape(F_SPECIES, 1).reinterpretAsFloats();
                            val = hiQ0.fma(d2Vec, negM2Vec).fma(that.getFloatVector(F_SPECIES, hiIdx), val);
                            val2 = hiQ1.fma(d2Vec, negM2Vec).fma(that.getFloatVector(F_SPECIES, hiIdx + F_SPECIES.length()), val2);
                        }
                        case 128 -> {
                            for (int p = 0; p < 4; p++) {
                                var loQ = loBytes.castShape(F_SPECIES, p).reinterpretAsFloats();
                                val = loQ.fma(d1Vec, negM1Vec).fma(that.getFloatVector(F_SPECIES, loIdx + p * F_SPECIES.length()), val);
                                var hiQ = hiBytes.castShape(F_SPECIES, p).reinterpretAsFloats();
                                val2 = hiQ.fma(d2Vec, negM2Vec).fma(that.getFloatVector(F_SPECIES, hiIdx + p * F_SPECIES.length()), val2);
                            }
                        }
                        default -> throw new UnsupportedOperationException(F_SPECIES.toString());
                    }
                }
            }
        }
        result += val.add(val2).reduceLanes(VectorOperators.ADD);

        // Handle tail
        if (j < size) {
            result += FloatTensor.scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j);
        }

        return result;
    }
}

final class Q5_KFloatTensor extends FloatTensor {

    static final int BLOCK_SIZE = GGMLType.QK_K;
    static final int TYPE_SIZE = GGMLType.Q5_K.getTypeSize();

    final long size;
    final MemorySegment memorySegment;

    public Q5_KFloatTensor(long size, MemorySegment memorySegment) {
        this.size = size;
        this.memorySegment = memorySegment;
    }

    @Override long size() { return size; }
    @Override public void setFloat(int index, float value) { throw new UnsupportedOperationException("setFloat"); }
    @Override FloatVector getFloatVector(VectorSpecies<Float> species, int index) { throw new UnsupportedOperationException("getFloatVector"); }
    @Override public GGMLType type() { return GGMLType.Q5_K; }

    @Override
    public float getFloat(long index) {
        long blockIndex = index / BLOCK_SIZE;
        int withinBlock = (int) (index % BLOCK_SIZE);
        long blockOffset = blockIndex * TYPE_SIZE;
        float d = readFloat16(memorySegment, blockOffset);
        float dmin = readFloat16(memorySegment, blockOffset + 2);
        long scalesOffset = blockOffset + 4;
        long qhOffset = blockOffset + 16;  // 4 + 12
        long qsOffset = blockOffset + 48;  // 4 + 12 + 32

        int group = withinBlock / 64;
        int inGroup = withinBlock % 64;
        boolean isHigh = inGroup >= 32;
        int l = isHigh ? inGroup - 32 : inGroup;
        int subBlock = isHigh ? group * 2 + 1 : group * 2;

        int sc = Q4_KFloatTensor.getScaleMinK4(subBlock, memorySegment, scalesOffset, false);
        int m = Q4_KFloatTensor.getScaleMinK4(subBlock, memorySegment, scalesOffset, true);

        byte qsByte = readByte(memorySegment, qsOffset + group * 32 + l);
        int nibble = isHigh ? ((Byte.toUnsignedInt(qsByte) >> 4) & 0xF) : (Byte.toUnsignedInt(qsByte) & 0xF);

        int qhBitPos = isHigh ? 2 * group + 1 : 2 * group;
        int qhBit = (Byte.toUnsignedInt(readByte(memorySegment, qhOffset + l)) >> qhBitPos) & 1;

        int quant = nibble | (qhBit << 4);
        return d * sc * quant - dmin * m;
    }

    @Override
    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        if (FloatTensor.USE_VECTOR_API) {
            return vectorDot(this, thisOffset, (ArrayFloatTensor) that, thatOffset, size);
        } else {
            return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
        }
    }

    private static float vectorDot(Q5_KFloatTensor thiz, int thisOffset, ArrayFloatTensor that, int thatOffset, int size) {
        float result = 0f;
        int j = 0;

        assert Integer.bitCount(BLOCK_SIZE) == 1 : "power of 2";
        int alignmentBound = Math.min(size, -thisOffset & (BLOCK_SIZE - 1));
        if (alignmentBound > 0) {
            result += FloatTensor.scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound);
            j += alignmentBound;
        }

        FloatVector val = FloatVector.zero(F_SPECIES);
        FloatVector val2 = FloatVector.zero(F_SPECIES);
        long blockOffset = (long) (thisOffset + j) / BLOCK_SIZE * TYPE_SIZE;
        int upperBound = j + (size - j) / BLOCK_SIZE * BLOCK_SIZE;

        for (; j < upperBound; j += BLOCK_SIZE, blockOffset += TYPE_SIZE) {
            float d = readFloat16(thiz.memorySegment, blockOffset);
            float dmin = readFloat16(thiz.memorySegment, blockOffset + 2);
            long scalesOff = blockOffset + 4;
            long qhOff = blockOffset + 16;
            long qsOff = blockOffset + 48;
            var qh0 = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, thiz.memorySegment, qhOff, ByteOrder.LITTLE_ENDIAN);
            var qh1 = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, thiz.memorySegment, qhOff + 16, ByteOrder.LITTLE_ENDIAN);

            for (int g = 0; g < 4; g++) {
                int loSubBlock = g * 2;
                int hiSubBlock = loSubBlock + 1;
                float d1 = d * Q4_KFloatTensor.getScaleMinK4(loSubBlock, thiz.memorySegment, scalesOff, false);
                float m1 = dmin * Q4_KFloatTensor.getScaleMinK4(loSubBlock, thiz.memorySegment, scalesOff, true);
                float d2 = d * Q4_KFloatTensor.getScaleMinK4(hiSubBlock, thiz.memorySegment, scalesOff, false);
                float m2 = dmin * Q4_KFloatTensor.getScaleMinK4(hiSubBlock, thiz.memorySegment, scalesOff, true);
                int qhBitPosLo = 2 * g;
                int qhBitPosHi = qhBitPosLo + 1;
                long groupQsOff = qsOff + (long) g * 32;
                var d1Vec = FloatVector.broadcast(F_SPECIES, d1);
                var d2Vec = FloatVector.broadcast(F_SPECIES, d2);
                var negM1Vec = FloatVector.broadcast(F_SPECIES, -m1);
                var negM2Vec = FloatVector.broadcast(F_SPECIES, -m2);

                for (int c = 0; c < 2; c++) {
                    int loBase = thatOffset + j + g * 64 + c * 16;
                    int hiBase = thatOffset + j + g * 64 + 32 + c * 16;

                    var wBytes = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, thiz.memorySegment,
                            groupQsOff + c * 16L, ByteOrder.LITTLE_ENDIAN);
                    var loQ = wBytes.and((byte) 0xF);
                    var hiQ = wBytes.lanewise(VectorOperators.LSHR, 4);

                    var qhBytes = c == 0 ? qh0 : qh1;
                    loQ = loQ.or(qhBytes.lanewise(VectorOperators.LSHR, qhBitPosLo).and((byte) 1).lanewise(VectorOperators.LSHL, 4));
                    hiQ = hiQ.or(qhBytes.lanewise(VectorOperators.LSHR, qhBitPosHi).and((byte) 1).lanewise(VectorOperators.LSHL, 4));

                    switch (F_SPECIES.vectorBitSize()) {
                        case 512 -> {
                            var loQf = loQ.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            var hiQf = hiQ.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            val = loQf.fma(d1Vec, negM1Vec).fma(that.getFloatVector(F_SPECIES, loBase), val);
                            val2 = hiQf.fma(d2Vec, negM2Vec).fma(that.getFloatVector(F_SPECIES, hiBase), val2);
                        }
                        case 256 -> {
                            var loQf0 = loQ.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            var loQf1 = loQ.castShape(F_SPECIES, 1).reinterpretAsFloats();
                            var hiQf0 = hiQ.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            var hiQf1 = hiQ.castShape(F_SPECIES, 1).reinterpretAsFloats();
                            val = loQf0.fma(d1Vec, negM1Vec).fma(that.getFloatVector(F_SPECIES, loBase), val);
                            val = loQf1.fma(d1Vec, negM1Vec).fma(that.getFloatVector(F_SPECIES, loBase + F_SPECIES.length()), val);
                            val2 = hiQf0.fma(d2Vec, negM2Vec).fma(that.getFloatVector(F_SPECIES, hiBase), val2);
                            val2 = hiQf1.fma(d2Vec, negM2Vec).fma(that.getFloatVector(F_SPECIES, hiBase + F_SPECIES.length()), val2);
                        }
                        case 128 -> {
                            for (int p = 0; p < 4; p++) {
                                int off = p * F_SPECIES.length();
                                var loQf = loQ.castShape(F_SPECIES, p).reinterpretAsFloats();
                                var hiQf = hiQ.castShape(F_SPECIES, p).reinterpretAsFloats();
                                val = loQf.fma(d1Vec, negM1Vec).fma(that.getFloatVector(F_SPECIES, loBase + off), val);
                                val2 = hiQf.fma(d2Vec, negM2Vec).fma(that.getFloatVector(F_SPECIES, hiBase + off), val2);
                            }
                        }
                        default -> throw new UnsupportedOperationException(F_SPECIES.toString());
                    }
                }
            }
        }

        result += val.add(val2).reduceLanes(VectorOperators.ADD);

        if (j < size) {
            result += FloatTensor.scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j);
        }

        return result;
    }
}

final class Q6_KFloatTensor extends FloatTensor {

    static final int BLOCK_SIZE = GGMLType.QK_K;
    static final int TYPE_SIZE = GGMLType.Q6_K.getTypeSize();

    final long size;
    final MemorySegment memorySegment;

    public Q6_KFloatTensor(long size, MemorySegment memorySegment) {
        this.size = size;
        this.memorySegment = memorySegment;
    }

    @Override long size() { return size; }
    @Override public void setFloat(int index, float value) { throw new UnsupportedOperationException("setFloat"); }
    @Override FloatVector getFloatVector(VectorSpecies<Float> species, int index) { throw new UnsupportedOperationException("getFloatVector"); }
    @Override public GGMLType type() { return GGMLType.Q6_K; }

    // Block layout: ql[128] | qh[64] | scales[16] (int8) | d (fp16)
    // 256 elements per block, 6-bit quants: 4 from ql nibble + 2 from qh

    @Override
    public float getFloat(long index) {
        long blockIndex = index / BLOCK_SIZE;
        int withinBlock = (int) (index % BLOCK_SIZE);
        long blockOffset = blockIndex * TYPE_SIZE;
        long qlOff = blockOffset;
        long qhOff = blockOffset + 128;
        long scOff = blockOffset + 192;
        float d = readFloat16(memorySegment, blockOffset + 208);

        int half = withinBlock / 128;
        int rem128 = withinBlock % 128;
        int sub32 = rem128 / 32;
        int l = rem128 % 32;

        long qlBase = qlOff + half * 64;
        long qhBase = qhOff + half * 32;

        int qlNibble, qhShift;
        switch (sub32) {
            case 0 -> { qlNibble = Byte.toUnsignedInt(readByte(memorySegment, qlBase + l)) & 0xF; qhShift = 0; }
            case 1 -> { qlNibble = Byte.toUnsignedInt(readByte(memorySegment, qlBase + 32 + l)) & 0xF; qhShift = 2; }
            case 2 -> { qlNibble = (Byte.toUnsignedInt(readByte(memorySegment, qlBase + l)) >> 4) & 0xF; qhShift = 4; }
            case 3 -> { qlNibble = (Byte.toUnsignedInt(readByte(memorySegment, qlBase + 32 + l)) >> 4) & 0xF; qhShift = 6; }
            default -> throw new IllegalStateException();
        }

        int qhBits = (Byte.toUnsignedInt(readByte(memorySegment, qhBase + l)) >> qhShift) & 3;
        int q6 = (qlNibble | (qhBits << 4)) - 32;
        int sc = readByte(memorySegment, scOff + half * 8 + sub32 * 2 + l / 16); // signed int8

        return d * sc * q6;
    }

    @Override
    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        if (FloatTensor.USE_VECTOR_API) {
            return vectorDot(this, thisOffset, (ArrayFloatTensor) that, thatOffset, size);
        } else {
            return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
        }
    }

    private static float fp16ToFloatNoIntrinsic(short h) {
        int bits = Short.toUnsignedInt(h);
        int sign = (bits & 0x8000) << 16;
        int exp = (bits >>> 10) & 0x1F;
        int mantissa = bits & 0x03FF;

        if (exp == 0) {
            if (mantissa == 0) {
                return Float.intBitsToFloat(sign);
            }
            int e = 127 - 15 + 1;
            while ((mantissa & 0x0400) == 0) {
                mantissa <<= 1;
                e--;
            }
            mantissa &= 0x03FF;
            return Float.intBitsToFloat(sign | (e << 23) | (mantissa << 13));
        }
        if (exp == 0x1F) {
            return Float.intBitsToFloat(sign | 0x7F80_0000 | (mantissa << 13));
        }
        return Float.intBitsToFloat(sign | ((exp + (127 - 15)) << 23) | (mantissa << 13));
    }

    private static float vectorDot(Q6_KFloatTensor thiz, int thisOffset, ArrayFloatTensor that, int thatOffset, int size) {
        float result = 0f;
        int j = 0;

        assert Integer.bitCount(BLOCK_SIZE) == 1 : "power of 2";
        int alignmentBound = Math.min(size, -thisOffset & (BLOCK_SIZE - 1));
        if (alignmentBound > 0) {
            result += FloatTensor.scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound);
            j += alignmentBound;
        }

        FloatVector acc = FloatVector.zero(F_SPECIES);
        long blockOffset = (long) (thisOffset + j) / BLOCK_SIZE * TYPE_SIZE;
        int upperBound = j + (size - j) / BLOCK_SIZE * BLOCK_SIZE;

        for (; j < upperBound; j += BLOCK_SIZE, blockOffset += TYPE_SIZE) {
            long qlOff = blockOffset;
            long qhOff = blockOffset + 128;
            long scOff = blockOffset + 192;
            // NOTE: Deliberately avoid Float.float16ToFloat here.
            // In native-image builds, Graal can lower that intrinsic to VCVTPH2PS with
            // an illegal high XMM operand under heavy vector register pressure in Q6_K
            // vectorDot, causing a compile-time crash. Keep this software conversion
            // until the Graal backend bug is fixed.
            float d = fp16ToFloatNoIntrinsic(readShort(thiz.memorySegment, blockOffset + 208));

            for (int h = 0; h < 2; h++) {
                long qlBase = qlOff + h * 64;
                long qhBase = qhOff + h * 32;

                int base = thatOffset + j + h * 128;
                for (int c = 0; c < 2; c++) {
                    var qlA = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, thiz.memorySegment,
                            qlBase + c * 16L, ByteOrder.LITTLE_ENDIAN);
                    var qlB = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, thiz.memorySegment,
                            qlBase + 32 + c * 16L, ByteOrder.LITTLE_ENDIAN);
                    var qhV = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, thiz.memorySegment,
                            qhBase + c * 16L, ByteOrder.LITTLE_ENDIAN);

                    var q0 = qlA.and((byte) 0xF).or(qhV.and((byte) 3).lanewise(VectorOperators.LSHL, 4)).sub((byte) 32);
                    var q1 = qlB.and((byte) 0xF).or(qhV.lanewise(VectorOperators.LSHR, 2).and((byte) 3).lanewise(VectorOperators.LSHL, 4)).sub((byte) 32);
                    var q2 = qlA.lanewise(VectorOperators.LSHR, 4).or(qhV.lanewise(VectorOperators.LSHR, 4).and((byte) 3).lanewise(VectorOperators.LSHL, 4)).sub((byte) 32);
                    var q3 = qlB.lanewise(VectorOperators.LSHR, 4).or(qhV.lanewise(VectorOperators.LSHR, 6).and((byte) 3).lanewise(VectorOperators.LSHL, 4)).sub((byte) 32);

                    float ds0 = d * readByte(thiz.memorySegment, scOff + h * 8 + c);
                    float ds1 = d * readByte(thiz.memorySegment, scOff + h * 8 + 2 + c);
                    float ds2 = d * readByte(thiz.memorySegment, scOff + h * 8 + 4 + c);
                    float ds3 = d * readByte(thiz.memorySegment, scOff + h * 8 + 6 + c);

                    var ds0Vec = FloatVector.broadcast(F_SPECIES, ds0);
                    var ds1Vec = FloatVector.broadcast(F_SPECIES, ds1);
                    var ds2Vec = FloatVector.broadcast(F_SPECIES, ds2);
                    var ds3Vec = FloatVector.broadcast(F_SPECIES, ds3);

                    int sg0Idx = base + c * 16;
                    int sg1Idx = base + 32 + c * 16;
                    int sg2Idx = base + 64 + c * 16;
                    int sg3Idx = base + 96 + c * 16;

                    switch (F_SPECIES.vectorBitSize()) {
                        case 512 -> {
                            var q0f = q0.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            var q1f = q1.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            var q2f = q2.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            var q3f = q3.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            acc = q0f.mul(ds0Vec).fma(that.getFloatVector(F_SPECIES, sg0Idx), acc);
                            acc = q1f.mul(ds1Vec).fma(that.getFloatVector(F_SPECIES, sg1Idx), acc);
                            acc = q2f.mul(ds2Vec).fma(that.getFloatVector(F_SPECIES, sg2Idx), acc);
                            acc = q3f.mul(ds3Vec).fma(that.getFloatVector(F_SPECIES, sg3Idx), acc);
                        }
                        case 256 -> {
                            for (int p = 0; p < 2; p++) {
                                int off = p * F_SPECIES.length();
                                var q0f = q0.castShape(F_SPECIES, p).reinterpretAsFloats();
                                var q1f = q1.castShape(F_SPECIES, p).reinterpretAsFloats();
                                var q2f = q2.castShape(F_SPECIES, p).reinterpretAsFloats();
                                var q3f = q3.castShape(F_SPECIES, p).reinterpretAsFloats();
                                acc = q0f.mul(ds0Vec).fma(that.getFloatVector(F_SPECIES, sg0Idx + off), acc);
                                acc = q1f.mul(ds1Vec).fma(that.getFloatVector(F_SPECIES, sg1Idx + off), acc);
                                acc = q2f.mul(ds2Vec).fma(that.getFloatVector(F_SPECIES, sg2Idx + off), acc);
                                acc = q3f.mul(ds3Vec).fma(that.getFloatVector(F_SPECIES, sg3Idx + off), acc);
                            }
                        }
                        case 128 -> {
                            for (int p = 0; p < 4; p++) {
                                int off = p * F_SPECIES.length();
                                var q0f = q0.castShape(F_SPECIES, p).reinterpretAsFloats();
                                var q1f = q1.castShape(F_SPECIES, p).reinterpretAsFloats();
                                var q2f = q2.castShape(F_SPECIES, p).reinterpretAsFloats();
                                var q3f = q3.castShape(F_SPECIES, p).reinterpretAsFloats();
                                acc = q0f.mul(ds0Vec).fma(that.getFloatVector(F_SPECIES, sg0Idx + off), acc);
                                acc = q1f.mul(ds1Vec).fma(that.getFloatVector(F_SPECIES, sg1Idx + off), acc);
                                acc = q2f.mul(ds2Vec).fma(that.getFloatVector(F_SPECIES, sg2Idx + off), acc);
                                acc = q3f.mul(ds3Vec).fma(that.getFloatVector(F_SPECIES, sg3Idx + off), acc);
                            }
                        }
                        default -> throw new UnsupportedOperationException(F_SPECIES.toString());
                    }
                }
            }
        }

        result += acc.reduceLanes(VectorOperators.ADD);

        if (j < size) {
            result += FloatTensor.scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j);
        }

        return result;
    }

}

final class Q8_0FloatTensor extends FloatTensor {

    final long size;
    final MemorySegment memorySegment;

    public Q8_0FloatTensor(long size, MemorySegment memorySegment) {
        this.size = size;
        this.memorySegment = memorySegment;
    }

    @Override
    long size() {
        return size;
    }

    @Override
    public void setFloat(int index, float value) {
        throw new UnsupportedOperationException("setFloat");
    }

    @Override
    FloatVector getFloatVector(VectorSpecies<Float> species, int index) {
        throw new UnsupportedOperationException("getFloatVector");
    }

    @Override
    public GGMLType type() {
        return GGMLType.Q8_0;
    }

    @Override
    public float getFloat(long index) {
        long blockIndex = index / GGMLType.Q8_0.getBlockSize();
        long withinBlockIndex = index % GGMLType.Q8_0.getBlockSize();
        long blockOffset = blockIndex * GGMLType.Q8_0.getTypeSize();
        byte quant = readByte(memorySegment, blockOffset + Float16.BYTES + withinBlockIndex);
        float scale = readFloat16(memorySegment, blockOffset);
        return quant * scale;
    }

    @Override
    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        if (FloatTensor.USE_VECTOR_API) {
            return vectorDot(this, thisOffset, (ArrayFloatTensor) that, thatOffset, size);
        } else {
            return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
        }
    }

    private static float vectorDot(Q8_0FloatTensor thiz, int thisOffset, ArrayFloatTensor that, int thatOffset, int size) {
        float result = 0f;
        int j = 0;

        assert Integer.bitCount(GGMLType.Q8_0.getBlockSize()) == 1 : "power of 2";
        int alignmentBound = Math.min(size, -thisOffset & (GGMLType.Q8_0.getBlockSize() - 1));
        if (alignmentBound > 0) {
            result += FloatTensor.scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound);
            j += alignmentBound;
        }
        assert (thisOffset + j) % GGMLType.Q8_0.getBlockSize() == 0;

        FloatVector val = FloatVector.zero(F_SPECIES);
        long blockOffset = (long) (thisOffset + j) / GGMLType.Q8_0.getBlockSize() * GGMLType.Q8_0.getTypeSize();
        int upperBound = j + (size - j) / GGMLType.Q8_0.getBlockSize() * GGMLType.Q8_0.getBlockSize();
        for (; j < upperBound; j += GGMLType.Q8_0.getBlockSize(), blockOffset += GGMLType.Q8_0.getTypeSize()) {
            float wScaleValue = readFloat16(thiz.memorySegment, blockOffset);
            var wScale = FloatVector.broadcast(F_SPECIES, wScaleValue);
            switch (F_SPECIES.vectorBitSize()) {
                case 512 -> {
                    var wBytes = ByteVector.fromMemorySegment(ByteVector.SPECIES_256, thiz.memorySegment, blockOffset + Float16.BYTES, ByteOrder.LITTLE_ENDIAN);
                    var s0 = that.getFloatVector(F_SPECIES, thatOffset + j).mul(wBytes.castShape(F_SPECIES, 0));
                    var s1 = that.getFloatVector(F_SPECIES, thatOffset + j + F_SPECIES.length()).mul(wBytes.castShape(F_SPECIES, 1));
                    val = s0.add(s1).fma(wScale, val);
                }
                case 256 -> {
                    var wBytes = ByteVector.fromMemorySegment(ByteVector.SPECIES_256, thiz.memorySegment, blockOffset + Float16.BYTES, ByteOrder.LITTLE_ENDIAN);
                    var s0 = that.getFloatVector(F_SPECIES, thatOffset + j).mul(wBytes.castShape(F_SPECIES, 0));
                    var s1 = that.getFloatVector(F_SPECIES, thatOffset + j + 2 * F_SPECIES.length()).mul(wBytes.castShape(F_SPECIES, 2));
                    s0 = that.getFloatVector(F_SPECIES, thatOffset + j + F_SPECIES.length()).fma(wBytes.castShape(F_SPECIES, 1), s0);
                    s1 = that.getFloatVector(F_SPECIES, thatOffset + j + 3 * F_SPECIES.length()).fma(wBytes.castShape(F_SPECIES, 3), s1);
                    val = s0.add(s1).fma(wScale, val);
                }
                case 128 -> {
                    for (int i = 0; i < 2; ++i) {
                        var wBytes = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, thiz.memorySegment, blockOffset + Float16.BYTES + i * ByteVector.SPECIES_128.vectorByteSize(), ByteOrder.LITTLE_ENDIAN);
                        var s0 = that.getFloatVector(F_SPECIES, thatOffset + j + i * 16).mul(wBytes.castShape(F_SPECIES, 0));
                        var s1 = that.getFloatVector(F_SPECIES, thatOffset + j + i * 16 + 2 * F_SPECIES.length()).mul(wBytes.castShape(F_SPECIES, 2));
                        s0 = that.getFloatVector(F_SPECIES, thatOffset + j + i * 16 + F_SPECIES.length()).fma(wBytes.castShape(F_SPECIES, 1), s0);
                        s1 = that.getFloatVector(F_SPECIES, thatOffset + j + i * 16 + 3 * F_SPECIES.length()).fma(wBytes.castShape(F_SPECIES, 3), s1);
                        val = s0.add(s1).fma(wScale, val);
                    }
                }
                default -> throw new UnsupportedOperationException(F_SPECIES.toString());
            }
        }
        result += val.reduceLanes(VectorOperators.ADD);

        if (j < size) {
            result += FloatTensor.scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j);
        }

        return result;
    }
}

final class MXFP4FloatTensor extends FloatTensor {

    private static final int[] MXFP4_VALUES = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

    private final long size;
    private final MemorySegment memorySegment;

    MXFP4FloatTensor(long size, MemorySegment memorySegment) {
        this.size = size;
        this.memorySegment = memorySegment;
    }

    @Override long size() { return size; }
    @Override public void setFloat(int index, float value) { throw new UnsupportedOperationException("setFloat"); }
    @Override FloatVector getFloatVector(VectorSpecies<Float> species, int index) { throw new UnsupportedOperationException("getFloatVector"); }
    @Override public GGMLType type() { return GGMLType.MXFP4; }

    @Override
    public float getFloat(long index) {
        assert 0 <= index && index < size;
        long blockIndex = index / GGMLType.QK_MXFP4;
        int inBlockIndex = (int) (index % GGMLType.QK_MXFP4);
        long blockOffset = blockIndex * GGMLType.MXFP4.getTypeSize();

        int e8m0 = Byte.toUnsignedInt(readByte(memorySegment, blockOffset));
        float d = e8m0ToFp32Half(e8m0);

        long qsOffset = blockOffset + Byte.BYTES + (inBlockIndex & 0x0F);
        int packed = Byte.toUnsignedInt(readByte(memorySegment, qsOffset));
        int nibble = inBlockIndex < (GGMLType.QK_MXFP4 / 2) ? (packed & 0x0F) : ((packed >>> 4) & 0x0F);

        return MXFP4_VALUES[nibble] * d;
    }

    @Override
    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        if (that instanceof ArrayFloatTensor aft) {
            if (FloatTensor.USE_VECTOR_API) {
                return vectorDot(this, thisOffset, aft, thatOffset, size);
            }
            return scalarDot(this, thisOffset, aft, thatOffset, size);
        }
        return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
    }

    private static float vectorDot(MXFP4FloatTensor thiz, int thisOffset, ArrayFloatTensor that, int thatOffset, int size) {
        assert Integer.bitCount(GGMLType.QK_MXFP4) == 1 : "power of 2";
        int j = 0;
        float result = 0f;

        int alignmentBound = Math.min(size, -thisOffset & (GGMLType.QK_MXFP4 - 1));
        if (alignmentBound > 0) {
            result += scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound);
            j = alignmentBound;
        }

        int upperBound = j + (size - j) / GGMLType.QK_MXFP4 * GGMLType.QK_MXFP4;
        for (; j < upperBound; j += GGMLType.QK_MXFP4) {
            assert (thisOffset + j) % GGMLType.QK_MXFP4 == 0;
            long blockOffset = (long) (thisOffset + j) / GGMLType.QK_MXFP4 * GGMLType.MXFP4.getTypeSize();
            float d = e8m0ToFp32Half(Byte.toUnsignedInt(readByte(thiz.memorySegment, blockOffset)));

            ByteVector packed = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, thiz.memorySegment, blockOffset + Byte.BYTES, ByteOrder.LITTLE_ENDIAN);
            ByteVector lo = packed.and((byte) 0x0F);
            ByteVector hi = packed.lanewise(VectorOperators.LSHR, 4);

            float blockSum = 0f;
            switch (F_SPECIES.vectorBitSize()) {
                case 512 -> {
                    FloatVector loCoeffs = mxfp4CodesToCoeffs((FloatVector) lo.castShape(F_SPECIES, 0));
                    FloatVector hiCoeffs = mxfp4CodesToCoeffs((FloatVector) hi.castShape(F_SPECIES, 0));
                    FloatVector xLo = that.getFloatVector(F_SPECIES, thatOffset + j);
                    FloatVector xHi = that.getFloatVector(F_SPECIES, thatOffset + j + GGMLType.QK_MXFP4 / 2);
                    blockSum += loCoeffs.fma(xLo, hiCoeffs.mul(xHi)).reduceLanes(VectorOperators.ADD);
                }
                case 256 -> {
                    FloatVector lo0 = mxfp4CodesToCoeffs((FloatVector) lo.castShape(F_SPECIES, 0));
                    FloatVector lo1 = mxfp4CodesToCoeffs((FloatVector) lo.castShape(F_SPECIES, 1));
                    FloatVector hi0 = mxfp4CodesToCoeffs((FloatVector) hi.castShape(F_SPECIES, 0));
                    FloatVector hi1 = mxfp4CodesToCoeffs((FloatVector) hi.castShape(F_SPECIES, 1));
                    FloatVector x0 = that.getFloatVector(F_SPECIES, thatOffset + j);
                    FloatVector x1 = that.getFloatVector(F_SPECIES, thatOffset + j + F_SPECIES.length());
                    FloatVector x2 = that.getFloatVector(F_SPECIES, thatOffset + j + GGMLType.QK_MXFP4 / 2);
                    FloatVector x3 = that.getFloatVector(F_SPECIES, thatOffset + j + GGMLType.QK_MXFP4 / 2 + F_SPECIES.length());
                    blockSum += lo0.fma(x0, lo1.mul(x1)).reduceLanes(VectorOperators.ADD);
                    blockSum += hi0.fma(x2, hi1.mul(x3)).reduceLanes(VectorOperators.ADD);
                }
                case 128 -> {
                    FloatVector sum = FloatVector.zero(F_SPECIES);
                    for (int p = 0; p < 4; p++) {
                        FloatVector loPart = mxfp4CodesToCoeffs((FloatVector) lo.castShape(F_SPECIES, p));
                        FloatVector hiPart = mxfp4CodesToCoeffs((FloatVector) hi.castShape(F_SPECIES, p));
                        FloatVector xLo = that.getFloatVector(F_SPECIES, thatOffset + j + p * F_SPECIES.length());
                        FloatVector xHi = that.getFloatVector(F_SPECIES, thatOffset + j + GGMLType.QK_MXFP4 / 2 + p * F_SPECIES.length());
                        sum = loPart.fma(xLo, sum);
                        sum = hiPart.fma(xHi, sum);
                    }
                    blockSum += sum.reduceLanes(VectorOperators.ADD);
                }
                default -> throw new UnsupportedOperationException(F_SPECIES.toString());
            }

            result += blockSum * d;
        }

        if (j < size) {
            result += scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j);
        }
        return result;
    }

    private static FloatVector mxfp4CodesToCoeffs(FloatVector codes) {
        FloatVector zero = FloatVector.zero(F_SPECIES);
        FloatVector eight = FloatVector.broadcast(F_SPECIES, 8f);
        var negMask = codes.compare(VectorOperators.GE, 8f);

        FloatVector t = codes.sub(zero.blend(eight, negMask));
        FloatVector mag = t
                .add(t.sub(4f).lanewise(VectorOperators.MAX, 0f))
                .add(t.sub(6f).lanewise(VectorOperators.MAX, 0f).mul(2f));
        return mag.blend(mag.neg(), negMask);
    }

    private static float scalarDot(MXFP4FloatTensor thiz, int thisOffset, ArrayFloatTensor that, int thatOffset, int size) {
        float result = 0f;
        for (int i = 0; i < size; i++) {
            result += thiz.getFloat(thisOffset + i) * that.values[thatOffset + i];
        }
        return result;
    }

    private static float e8m0ToFp32Half(int x) {
        int bits;
        if (x < 2) {
            bits = 0x00200000 << x;
        } else {
            bits = (x - 1) << 23;
        }
        return Float.intBitsToFloat(bits);
    }
}

final class BF16FloatTensor extends FloatTensor {

    final long size;
    final MemorySegment memorySegment;

    public BF16FloatTensor(long size, MemorySegment memorySegment) {
        this.size = size;
        this.memorySegment = memorySegment;
    }

    @Override long size() { return size; }
    @Override public void setFloat(int index, float value) { throw new UnsupportedOperationException("setFloat"); }
    @Override FloatVector getFloatVector(VectorSpecies<Float> species, int index) { throw new UnsupportedOperationException("getFloatVector"); }
    @Override public GGMLType type() { return GGMLType.BF16; }

    @Override
    public float getFloat(long index) {
        assert 0 <= index && index < size;
        short bits = readShort(memorySegment, (long) index * 2);
        return Float.intBitsToFloat(bits << 16);
    }

    @Override
    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        if (FloatTensor.USE_VECTOR_API) {
            return vectorDot(this, thisOffset, (ArrayFloatTensor) that, thatOffset, size);
        } else {
            return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
        }
    }

    private static float vectorDot(BF16FloatTensor thiz, int thisOffset, ArrayFloatTensor that, int thatOffset, int size) {
        assert S_SPECIES_HALF.length() == F_SPECIES.length();
        FloatVector val = FloatVector.zero(F_SPECIES);
        int upperBound = F_SPECIES.loopBound(size);
        for (int i = 0; i < upperBound; i += F_SPECIES.length()) {
            FloatVector thatVector = that.getFloatVector(F_SPECIES, thatOffset + i);
            ShortVector bfloat16 = ShortVector.fromMemorySegment(S_SPECIES_HALF, thiz.memorySegment, (thisOffset + i) * 2L, ByteOrder.LITTLE_ENDIAN);
            FloatVector thizVector = bfloat16
                    .castShape(I_SPECIES, 0)
                    .lanewise(VectorOperators.LSHL, 16)
                    .reinterpretAsFloats();
            val = thizVector.fma(thatVector, val);
        }
        float result = val.reduceLanes(VectorOperators.ADD);
        if (upperBound < size) {
            result += scalarDot(thiz, thisOffset + upperBound, that, thatOffset + upperBound, size - upperBound);
        }
        return result;
    }
}

final class F16FloatTensor extends FloatTensor {

    final long size;
    final MemorySegment memorySegment;

    public F16FloatTensor(long size, MemorySegment memorySegment) {
        this.size = size;
        this.memorySegment = memorySegment;
    }

    static F16FloatTensor allocate(int... dims) {
        int n = FloatTensor.numberOfElements(dims);
        MemorySegment segment = Arena.ofAuto().allocate((long) n * 2);
        return new F16FloatTensor(n, segment);
    }

    @Override long size() { return size; }
    @Override FloatVector getFloatVector(VectorSpecies<Float> species, int index) { throw new UnsupportedOperationException("getFloatVector"); }
    @Override public GGMLType type() { return GGMLType.F16; }

    static FloatVector f16ToF32Vector(MemorySegment memSeg, long byteOffset) {
        ShortVector bits16 = ShortVector.fromMemorySegment(S_SPECIES_HALF, memSeg, byteOffset, ByteOrder.LITTLE_ENDIAN);
        var bits32 = bits16.castShape(I_SPECIES, 0).reinterpretAsInts();
        var zeroExponentMask = bits32.and(0x7C00).neg().lanewise(VectorOperators.ASHR, 31);
        bits32 = bits32.and(0x8000).lanewise(VectorOperators.LSHL, 16)
                .or(bits32.and(0x7FFF).add(0x1C000).lanewise(VectorOperators.LSHL, 13).and(zeroExponentMask));
        return bits32.reinterpretAsFloats();
    }

    @Override
    public float getFloat(long index) {
        assert 0 <= index && index < size;
        return readFloat16(memorySegment, index * 2);
    }

    @Override
    public void setFloat(int index, float value) {
        writeShort(memorySegment, (long) index * 2, Float.floatToFloat16(value));
    }

    @Override
    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        if (FloatTensor.USE_VECTOR_API) {
            return vectorDot(this, thisOffset, (ArrayFloatTensor) that, thatOffset, size);
        }
        return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
    }

    private static float vectorDot(F16FloatTensor thiz, int thisOffset, ArrayFloatTensor that, int thatOffset, int size) {
        assert S_SPECIES_HALF.length() == F_SPECIES.length();
        FloatVector val = FloatVector.zero(F_SPECIES);
        int upperBound = F_SPECIES.loopBound(size);
        for (int i = 0; i < upperBound; i += F_SPECIES.length()) {
            FloatVector thatVector = that.getFloatVector(F_SPECIES, thatOffset + i);
            FloatVector thizVector = f16ToF32Vector(thiz.memorySegment, (thisOffset + i) * 2L);
            val = thizVector.fma(thatVector, val);
        }
        float result = val.reduceLanes(VectorOperators.ADD);
        if (upperBound < size) {
            result += scalarDot(thiz, thisOffset + upperBound, that, thatOffset + upperBound, size - upperBound);
        }
        return result;
    }
}

final class ArrayFloatTensor extends FloatTensor {

    final long size;
    final float[] values;

    ArrayFloatTensor(float[] values) {
        this.size = values.length;
        this.values = values;
    }

    ArrayFloatTensor(FloatBuffer buf) {
        this.values = new float[buf.remaining()];
        this.size = values.length;
        buf.get(this.values);
        buf.rewind();
    }

    public static FloatTensor allocate(int... dims) {
        int numberOfElements = FloatTensor.numberOfElements(dims);
        return new ArrayFloatTensor(new float[numberOfElements]);
    }

    @Override
    public long size() {
        return size;
    }

    @Override
    public float getFloat(long index) {
        return values[Math.toIntExact(index)];
    }

    @Override
    public void setFloat(int index, float value) {
        values[index] = value;
    }

    @Override
    public GGMLType type() {
        return GGMLType.F32;
    }

    @Override
    public FloatTensor fillInPlace(int thisOffset, int size, float value) {
        Arrays.fill(values, thisOffset, thisOffset + size, value);
        return this;
    }

    @Override
    public FloatVector getFloatVector(VectorSpecies<Float> species, int index) {
        if (!USE_VECTOR_API) {
            throw new UnsupportedOperationException();
        }
        return FloatVector.fromArray(species, values, index);
    }

    @Override
    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        if (that instanceof ArrayFloatTensor aft) {
            if (USE_VECTOR_API) {
                return vectorDot(this, thisOffset, aft, thatOffset, size);
            }
            return FloatTensor.scalarDot(this, thisOffset, aft, thatOffset, size);
        }
        return that.dot(thatOffset, this, thisOffset, size);
    }

    private static float vectorDot(ArrayFloatTensor thiz, int thisOffset, ArrayFloatTensor that, int thatOffset, int size) {
        FloatVector val = FloatVector.zero(F_SPECIES);
        int upperBound = F_SPECIES.loopBound(size);
        for (int i = 0; i < upperBound; i += F_SPECIES.length()) {
            var a = FloatVector.fromArray(F_SPECIES, thiz.values, thisOffset + i);
            var b = FloatVector.fromArray(F_SPECIES, that.values, thatOffset + i);
            val = a.fma(b, val);
        }
        float result = val.reduceLanes(VectorOperators.ADD);
        for (int i = upperBound; i < size; i++) {
            result += thiz.values[thisOffset + i] * that.values[thatOffset + i];
        }
        return result;
    }

    @Override
    FloatTensor saxpyInPlace(int thisOffset, FloatTensor that, int thatOffset, int size, float a) {
        if (that instanceof F16FloatTensor f16 && USE_VECTOR_API) {
            return vectorSaxpyF16(this, thisOffset, f16, thatOffset, size, a);
        }
        return super.saxpyInPlace(thisOffset, that, thatOffset, size, a);
    }

    private static FloatTensor vectorSaxpyF16(ArrayFloatTensor thiz, int thisOffset,
                                               F16FloatTensor that, int thatOffset,
                                               int size, float a) {
        FloatVector va = FloatVector.broadcast(F_SPECIES, a);
        int upperBound = F_SPECIES.loopBound(size);
        for (int i = 0; i < upperBound; i += F_SPECIES.length()) {
            FloatVector thatVector = F16FloatTensor.f16ToF32Vector(that.memorySegment, (thatOffset + i) * 2L);
            FloatVector thisVector = FloatVector.fromArray(F_SPECIES, thiz.values, thisOffset + i);
            va.fma(thatVector, thisVector).intoArray(thiz.values, thisOffset + i);
        }
        for (int i = upperBound; i < size; i++) {
            thiz.values[thisOffset + i] += a * that.getFloat(thatOffset + i);
        }
        return thiz;
    }
}

final class F32FloatTensor extends FloatTensor {

    private final long size;
    private final MemorySegment memorySegment;

    F32FloatTensor(long numElements, MemorySegment memorySegment) {
        this.size = numElements;
        this.memorySegment = memorySegment;
    }

    @Override public long size() { return size; }

    @Override
    public float getFloat(long index) {
        return memorySegment.get(ValueLayout.JAVA_FLOAT_UNALIGNED, (long) index * Float.BYTES);
    }

    @Override
    public void setFloat(int index, float value) {
        throw new UnsupportedOperationException("read-only");
    }

    @Override public GGMLType type() { return GGMLType.F32; }

    @Override
    public FloatVector getFloatVector(VectorSpecies<Float> species, int index) {
        if (!USE_VECTOR_API) {
            throw new UnsupportedOperationException();
        }
        return FloatVector.fromMemorySegment(species, memorySegment, (long) index * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
    }

    @Override
    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        if (that instanceof ArrayFloatTensor aft && USE_VECTOR_API) {
            return vectorDot(this, thisOffset, aft, thatOffset, size);
        }
        return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
    }

    private static float vectorDot(F32FloatTensor thiz, int thisOffset, ArrayFloatTensor that, int thatOffset, int size) {
        FloatVector val = FloatVector.zero(F_SPECIES);
        int upperBound = F_SPECIES.loopBound(size);
        for (int i = 0; i < upperBound; i += F_SPECIES.length()) {
            var a = FloatVector.fromMemorySegment(F_SPECIES, thiz.memorySegment, (long) (thisOffset + i) * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
            var b = FloatVector.fromArray(F_SPECIES, that.values, thatOffset + i);
            val = a.fma(b, val);
        }
        float result = val.reduceLanes(VectorOperators.ADD);
        for (int i = upperBound; i < size; i++) {
            result += thiz.getFloat(thisOffset + i) * that.values[thatOffset + i];
        }
        return result;
    }
}

final class RoPE {
    public static Pair<float[], float[]> precomputeFreqsCis(int contextLength, int headSize, double theta) {
        assert headSize % 2 == 0;
        int halfHead = headSize / 2;
        float[] cr = new float[contextLength * halfHead];
        float[] ci = new float[contextLength * halfHead];
        int n = 0;
        for (int pos = 0; pos < contextLength; ++pos) {
            for (int i = 0; i < headSize; i += 2) {
                float freq = (float) (1.0 / Math.pow(theta, i / (double) headSize));
                float val = pos * freq;
                cr[n] = (float) Math.cos(val);
                ci[n] = (float) Math.sin(val);
                n++;
            }
        }
        assert contextLength * halfHead == n;
        return new Pair<>(cr, ci);
    }

    public static Pair<float[], float[]> precomputeFreqsCisFromFreqs(int contextLength, int headSize, double ropeTheta, float[] ropeFreqFactors) {
        // freq_factors are divisors on top of the standard RoPE base frequencies:
        // theta_i = pos * (1 / (ropeTheta^(2i/headSize))) / freqFactors[i]
        int halfHead = ropeFreqFactors.length;
        assert halfHead == headSize / 2;
        float[] cr = new float[contextLength * halfHead];
        float[] ci = new float[contextLength * halfHead];
        int n = 0;
        for (int pos = 0; pos < contextLength; ++pos) {
            for (int i = 0; i < halfHead; i++) {
                float baseFreq = (float) (1.0 / Math.pow(ropeTheta, (2.0 * i) / headSize));
                float val = pos * baseFreq / ropeFreqFactors[i];
                cr[n] = (float) Math.cos(val);
                ci[n] = (float) Math.sin(val);
                n++;
            }
        }
        assert contextLength * halfHead == n;
        return new Pair<>(cr, ci);
    }
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

@FunctionalInterface
interface Sampler {
    int sampleToken(FloatTensor logits);

    Sampler ARGMAX = FloatTensor::argmax;
}

record CategoricalSampler(RandomGenerator rng) implements Sampler {

    @Override
    public int sampleToken(FloatTensor logits) {
        float random0to1 = rng.nextFloat(1f);
        float cdf = 0.0f;
        for (int i = 0; i < logits.size(); i++) {
            cdf += logits.getFloat(i);
            if (random0to1 < cdf) {
                return i;
            }
        }
        return Math.toIntExact(logits.size()) - 1;
    }
}

final class ToppSampler implements Sampler {

    final int[] indices;
    final float topp;
    final RandomGenerator rng;

    public ToppSampler(int maxNumberOfElements, float topp, RandomGenerator rng) {
        this.indices = new int[maxNumberOfElements];
        this.topp = topp;
        this.rng = rng;
    }

    static void swap(int[] array, int from, int to) {
        int tmp = array[from];
        array[from] = array[to];
        array[to] = tmp;
    }

    static void siftDown(int[] array, int from, int n, Comparator<Integer> comparator) {
        int prev = from, next;
        while ((next = 2 * prev + 1) < n) {
            int r = 2 * prev + 2;
            if (r < n && comparator.compare(array[r], array[next]) < 0) {
                next = r;
            }
            if (comparator.compare(array[next], array[prev]) < 0) {
                swap(array, prev, next);
                prev = next;
            } else {
                break;
            }
        }
    }

    @Override
    public int sampleToken(FloatTensor logits) {
        Comparator<Integer> comparator = Comparator.comparingDouble((Integer i) -> logits.getFloat(i)).reversed();

        int n = Math.toIntExact(logits.size());
        int head = 0;
        int tail = n - 1;
        float cutoff = (1.0f - topp) / (n - 1);
        for (int i = 0; i < indices.length; i++) {
            if (logits.getFloat(i) >= cutoff) {
                indices[head++] = i;
            } else {
                indices[tail--] = i;
            }
        }

        int n0 = head;
        for (int i = n0 / 2 - 1; i >= 0; --i) {
            siftDown(indices, i, n0, comparator);
        }

        float cumulativeProb = 0.0f;
        int lastIndex = 0;
        for (int i = n0 - 1; i >= 0; i--) {
            swap(indices, 0, i);
            cumulativeProb += logits.getFloat(indices[i]);
            if (cumulativeProb > topp) {
                lastIndex = i;
                break;
            }
            siftDown(indices, 0, i, comparator);
        }

        float r = rng.nextFloat(1f) * cumulativeProb;
        float cdf = 0.0f;
        for (int i = n0 - 1; i >= lastIndex; i--) {
            cdf += logits.getFloat(indices[i]);
            if (r < cdf) {
                return indices[i];
            }
        }

        return indices[lastIndex];
    }
}

class LFMChatFormat {

    protected final LFMTokenizer tokenizer;
    protected final int beginOfSentence;
    protected final int startOfTurn;
    protected final int endOfTurn;
    protected final int endOfSentence;
    protected final int fimSuffix;
    protected final int fimPrefix;
    protected final int fimMiddle;
    protected final int fileSeparator;
    private final Set<Integer> stopTokens;

    public LFMChatFormat(LFMTokenizer tokenizer) {
        this.tokenizer = tokenizer;
        Map<String, Integer> specialTokens = this.tokenizer.getSpecialTokens();
        this.beginOfSentence = specialTokens.getOrDefault("<bos>", specialTokens.getOrDefault("<|startoftext|>", 1));
        this.startOfTurn = specialTokens.getOrDefault("<|im_start|>", specialTokens.getOrDefault("<|turn>", beginOfSentence));
        this.endOfTurn = specialTokens.getOrDefault("<|im_end|>", specialTokens.getOrDefault("<turn|>", -1));
        this.endOfSentence = specialTokens.getOrDefault("<eos>", specialTokens.getOrDefault("<|endoftext|>", 2));

        this.fimSuffix = specialTokens.getOrDefault("<|fim_suffix|>", -1);
        this.fimPrefix = specialTokens.getOrDefault("<|fim_prefix|>", -1);
        this.fimMiddle = specialTokens.getOrDefault("<|fim_middle|>", -1);
        this.fileSeparator = specialTokens.getOrDefault("<|file_separator|>", -1);

        Set<Integer> tokens = new HashSet<>();
        tokens.add(endOfSentence);
        if (endOfTurn >= 0) tokens.add(endOfTurn);
        if (fimSuffix != -1) tokens.add(fimSuffix);
        if (fimPrefix != -1) tokens.add(fimPrefix);
        if (fimMiddle != -1) tokens.add(fimMiddle);
        if (fileSeparator != -1) tokens.add(fileSeparator);
        this.stopTokens = Collections.unmodifiableSet(tokens);
    }

    public Set<Integer> getStopTokens() {
        return stopTokens;
    }

    public List<Integer> encodeHeader(LFMChatFormat.Message message) {
        List<Integer> tokens = new ArrayList<>();
        tokens.add(startOfTurn);
        tokens.addAll(tokenizer.encode(message.role().toString()));
        tokens.addAll(this.tokenizer.encode("\n"));
        return tokens;
    }

    public List<Integer> encodeMessage(LFMChatFormat.Message message) {
        List<Integer> tokens = this.encodeHeader(message);
        tokens.addAll(this.tokenizer.encode(message.content().strip()));
        if (endOfTurn >= 0) tokens.add(endOfTurn);
        tokens.addAll(this.tokenizer.encode("\n"));
        return tokens;
    }

    public List<Integer> encodeSystemThinkingTurn(String systemPrompt) {
        return encodeMessage(new Message(Role.SYSTEM, systemPrompt == null ? "" : systemPrompt));
    }

    public List<Integer> encodeGenerationPrompt() {
        return encodeHeader(new Message(Role.ASSISTANT, ""));
    }

    public record Message(LFMChatFormat.Role role, String content) {
    }

    public List<Integer> encodeFillInTheMiddle(String prefix, String suffix) {
        List<Integer> tokens = new ArrayList<>();
        tokens.add(this.fimPrefix);
        tokens.addAll(tokenizer.encode(prefix));
        tokens.add(this.fimSuffix);
        tokens.addAll(tokenizer.encode(suffix));
        tokens.add(this.fimMiddle);
        return tokens;
    }

    public record Role(String name) {
        public static LFMChatFormat.Role SYSTEM = new LFMChatFormat.Role("system");
        public static LFMChatFormat.Role USER = new LFMChatFormat.Role("user");
        public static LFMChatFormat.Role ASSISTANT = new LFMChatFormat.Role("assistant");

        @Override
        public String toString() {
            return name;
        }
    }
}

public class LFM25 {

    static Sampler selectSampler(int vocabularySize, float temperature, float topp, long rngSeed) {
        Sampler sampler;
        if (temperature == 0.0f) {
            sampler = Sampler.ARGMAX;
        } else {
            RandomGenerator rng = RandomGeneratorFactory.getDefault().create(rngSeed);
            Sampler innerSampler;
            if (topp <= 0 || topp >= 1) {
                innerSampler = new CategoricalSampler(rng);
            } else {
                innerSampler = new ToppSampler(vocabularySize, topp, rng);
            }
            sampler = logits -> {
                int logitsSize = Math.toIntExact(logits.size());
                logits.divideInPlace(0, logitsSize, temperature);
                logits.softmaxInPlace(0, logitsSize);
                return innerSampler.sampleToken(logits);
            };
        }
        return sampler;
    }

    private static final String ANSI_GREY  = "\033[90m";
    private static final String ANSI_RESET = "\033[0m";

    private static IntConsumer plainStreamingPrinter(LFMTokenizer tokenizer) {
        return token -> {
            if (!tokenizer.isSpecialToken(token)) {
                byte[] bytes = tokenizer.decodeTokenBytes(token);
                System.out.write(bytes, 0, bytes.length);
            }
        };
    }

    private static void onThinkingStart(PrintStream thoughtOut, boolean ansi) {
        if (ansi) {
            thoughtOut.print(ANSI_GREY);
        }
        thoughtOut.println("[Start thinking]");
    }

    private static void onThinkingEnd(PrintStream thoughtOut, boolean ansi, boolean emitted) {
        if (emitted) {
            thoughtOut.println();
        }
        thoughtOut.println("[End thinking]");
        if (ansi) {
            thoughtOut.print(ANSI_RESET);
        }
        thoughtOut.println();
    }

    static boolean supportsAnsiColors(String colorMode) {
        return switch (colorMode) {
            case "on" -> true;
            case "off" -> false;
            case "auto" -> {
                if (System.console() == null) {
                    yield false;
                }
                String noColor = System.getenv("NO_COLOR");
                if (noColor != null) {
                    yield false;
                }
                String term = System.getenv("TERM");
                yield term == null || !"dumb".equalsIgnoreCase(term);
            }
            default -> false;
        };
    }

    private static IntConsumer streamingPrinter(LFMTokenizer tokenizer, Options options) {
        if (!options.stream()) {
            return token -> {};
        }

        Integer thinkOpen = tokenizer.getSpecialTokens().get("<think>");
        Integer thinkClose = tokenizer.getSpecialTokens().get("</think>");
        if (thinkOpen == null || thinkClose == null) {
            return plainStreamingPrinter(tokenizer);
        }

        boolean thinkEnabled = options.think();
        PrintStream thoughtOut = options.thinkInline() ? System.out : System.err;
        boolean ansi = options.colors();
        boolean[] inThink = {false};
        boolean[] emitted = {false};
        return token -> {
            if (token == thinkOpen) {
                if (thinkEnabled) {
                    onThinkingStart(thoughtOut, ansi);
                }
                inThink[0] = true;
                emitted[0] = false;
                return;
            }
            if (token == thinkClose) {
                if (thinkEnabled) {
                    onThinkingEnd(thoughtOut, ansi, emitted[0]);
                }
                inThink[0] = false;
                emitted[0] = false;
                return;
            }
            if (!tokenizer.isSpecialToken(token)) {
                byte[] bytes = tokenizer.decodeTokenBytes(token);
                if (inThink[0]) {
                    if (thinkEnabled) {
                        thoughtOut.write(bytes, 0, bytes.length);
                        emitted[0] = true;
                    }
                } else {
                    System.out.write(bytes, 0, bytes.length);
                }
            }
        };
    }

    private static List<Integer> visibleTokens(LFMTokenizer tokenizer, List<Integer> tokens, boolean think) {
        if (think) {
            return tokens;
        }
        return stripThoughtTokens(tokenizer, tokens);
    }

    static void appendThinkSurrogate(LFMTokenizer tokenizer, List<Integer> tokens) {
        Integer start = tokenizer.getSpecialTokens().get("<think>");
        Integer end = tokenizer.getSpecialTokens().get("</think>");
        if (start == null || end == null) return;
        List<Integer> nl = tokenizer.encode("\n");
        tokens.add(start);
        tokens.addAll(nl);
        tokens.add(end);
        tokens.addAll(nl);
        tokens.addAll(nl);
    }

    private static List<Integer> stripThoughtTokens(LFMTokenizer tokenizer, List<Integer> tokens) {
        Integer thinkOpen = tokenizer.getSpecialTokens().get("<think>");
        Integer thinkClose = tokenizer.getSpecialTokens().get("</think>");
        if (thinkOpen == null || thinkClose == null) {
            return tokens;
        }
        List<Integer> result = new ArrayList<>();
        boolean inThink = false;
        boolean sawOpen = false;
        boolean sawClose = false;
        for (int token : tokens) {
            if (token == thinkOpen) {
                inThink = true;
                sawOpen = true;
                continue;
            }
            if (token == thinkClose) {
                inThink = false;
                sawClose = true;
                continue;
            }
            if (!inThink) {
                result.add(token);
            }
        }
        // If we saw <think> but no </think>, model didn't close the tag;
        // fall back to showing everything rather than hiding it all
        if (sawOpen && !sawClose) {
            return tokens;
        }
        return result;
    }

    static void runInteractive(Llama model, Sampler sampler, Options options) throws IOException {
        Llama.State state = null;
        LFMChatFormat chatFormat = new LFMChatFormat(model.tokenizer());
        List<Integer> conversationTokens = new ArrayList<>();
        conversationTokens.add(chatFormat.beginOfSentence);
        if (options.systemPrompt() != null) {
            conversationTokens.addAll(chatFormat.encodeSystemThinkingTurn(options.systemPrompt()));
        }
        int startPosition = 0;
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(System.in))) {
            while (true) {
                System.out.print("> ");
                System.out.flush();
                String userText = reader.readLine();
                if (userText == null) break;
                switch (userText) {
                    case "/quit", "/exit" -> { return; }
                    case "/context" -> {
                        System.out.printf("%d out of %d context tokens used (%d tokens remaining)%n",
                                conversationTokens.size(),
                                options.maxTokens(),
                                options.maxTokens() - conversationTokens.size());
                        continue;
                    }
                }
            if (state == null) {
                state = model.createNewState();
            }
            conversationTokens.addAll(chatFormat.encodeMessage(new LFMChatFormat.Message(LFMChatFormat.Role.USER, userText)));
            conversationTokens.addAll(chatFormat.encodeGenerationPrompt());
            if (!options.think()) {
                appendThinkSurrogate(model.tokenizer(), conversationTokens);
            }

            Set<Integer> stopTokens = chatFormat.getStopTokens();
            IntConsumer printer = streamingPrinter(model.tokenizer(), options);
            List<Integer> responseTokens = Llama.generateTokens(model, state, startPosition, conversationTokens.subList(startPosition, conversationTokens.size()), stopTokens, options.maxTokens(), sampler, options.echo(), options.colors(), printer);
            Integer stopToken = null;
            if (!responseTokens.isEmpty() && stopTokens.contains(responseTokens.getLast())) {
                stopToken = responseTokens.getLast();
                responseTokens.removeLast();
            }
            List<Integer> visibleResponseTokens = visibleTokens(model.tokenizer(), responseTokens, options.think());
            conversationTokens.addAll(options.keepPastThinking() ? responseTokens : visibleResponseTokens);
            if (stopToken != null) {
                conversationTokens.add(stopToken);
                if (stopToken == chatFormat.endOfTurn) {
                    conversationTokens.addAll(model.tokenizer().encode("\n"));
                }
            }
            startPosition = conversationTokens.size();
            if (!options.stream()) {
                String responseText = model.tokenizer().decode(visibleResponseTokens);
                System.out.println(responseText);
            }
            if (stopToken == null) {
                System.err.println("Ran out of context length...");
                break;
            }
            }
        }
    }

    static void runInstructOnce(Llama model, Sampler sampler, Options options) {
        Llama.State state = model.createNewState();
        LFMChatFormat chatFormat = new LFMChatFormat(model.tokenizer());
        List<Integer> promptTokens = new ArrayList<>();
        if (options.rawPrompt()) {
            promptTokens.addAll(encodeWithSpecialTokens(model.tokenizer(), options.prompt()));
        } else {
            promptTokens.add(chatFormat.beginOfSentence);
            if (options.suffix() != null) {
                promptTokens.addAll(chatFormat.encodeFillInTheMiddle(options.prompt(), options.suffix()));
            } else {
                if (options.systemPrompt() != null) {
                    promptTokens.addAll(chatFormat.encodeSystemThinkingTurn(options.systemPrompt()));
                }
                promptTokens.addAll(chatFormat.encodeMessage(new LFMChatFormat.Message(LFMChatFormat.Role.USER, options.prompt())));
                promptTokens.addAll(chatFormat.encodeGenerationPrompt());
                if (!options.think()) {
                    appendThinkSurrogate(model.tokenizer(), promptTokens);
                }
            }
        }

        Set<Integer> stopTokens = chatFormat.getStopTokens();
        IntConsumer printer = streamingPrinter(model.tokenizer(), options);
        List<Integer> responseTokens = Llama.generateTokens(model, state, 0, promptTokens, stopTokens, options.maxTokens(), sampler, options.echo(), options.colors(), printer);
        if (!responseTokens.isEmpty() && stopTokens.contains(responseTokens.getLast())) {
            responseTokens.removeLast();
        }
        List<Integer> visibleResponseTokens = visibleTokens(model.tokenizer(), responseTokens, options.think());
        if (!options.stream()) {
            String responseText = model.tokenizer().decode(visibleResponseTokens);
            System.out.println(responseText);
        }
    }

    static List<Integer> encodeWithSpecialTokens(LFMTokenizer tokenizer, String text) {
        Map<String, Integer> special = tokenizer.getSpecialTokens();
        if (special.isEmpty()) {
            return tokenizer.encode(text);
        }
        List<String> keys = new ArrayList<>(special.keySet());
        keys.sort((a, b) -> Integer.compare(b.length(), a.length()));

        List<Integer> out = new ArrayList<>();
        StringBuilder ordinary = new StringBuilder();
        int i = 0;
        while (i < text.length()) {
            String matched = null;
            for (String k : keys) {
                if (text.startsWith(k, i)) {
                    matched = k;
                    break;
                }
            }
            if (matched != null) {
                if (!ordinary.isEmpty()) {
                    out.addAll(tokenizer.encode(ordinary.toString()));
                    ordinary.setLength(0);
                }
                out.add(special.get(matched));
                i += matched.length();
            } else {
                ordinary.append(text.charAt(i));
                i += 1;
            }
        }
        if (!ordinary.isEmpty()) {
            out.addAll(tokenizer.encode(ordinary.toString()));
        }
        return out;
    }

    static final int DEFAULT_MAX_TOKENS = 1024;

    record Options(Path modelPath, String prompt, String suffix, String systemPrompt, boolean interactive,
                    float temperature, float topp, long seed, int maxTokens, boolean stream, boolean echo,
                    boolean think, boolean thinkInline, boolean colors,
                    boolean keepPastThinking, boolean rawPrompt) {

        Options {
            require(modelPath != null, "Missing argument: --model <path> is required");
            require(interactive || prompt != null, "Missing argument: --prompt is required in --instruct mode e.g. --prompt \"Why is the sky blue?\"");
            require(0 <= temperature, "Invalid argument: --temperature must be non-negative");
            require(0 <= topp && topp <= 1, "Invalid argument: --top-p must be within [0, 1]");
        }

        static void require(boolean condition, String messageFormat, Object... args) {
            if (!condition) {
                throw new IllegalArgumentException("ERROR " + messageFormat.formatted(args));
            }
        }

        static boolean parseBooleanOption(String optionName, String value) {
            return switch (value.toLowerCase(Locale.ROOT)) {
                case "true", "on" -> true;
                case "false", "off" -> false;
                default -> {
                    require(false, "Invalid argument for %s: expected true|false|on|off, got %s", optionName, value);
                    yield false;
                }
            };
        }

        static void printUsage(PrintStream out) {
            out.println("Usage:  jbang LFM25.java [options]");
            out.println();
            out.println("Options:");
            out.println("  --model, -m <path>            required, path to .gguf file");
            out.println("  --interactive, --chat, -i     run in chat mode");
            out.println("  --instruct                    run in instruct (once) mode, default mode");
            out.println("  --prompt, -p <string>         input prompt");
            out.println("  --suffix <string>             suffix for fill-in-the-middle request");
            out.println("  --system-prompt, -sp <string> system prompt for chat/instruct mode");
            out.println("  --temperature, -temp <float>  temperature in [0,inf], default 1.0");
            out.println("  --top-p <float>               p value in top-p (nucleus) sampling in [0,1] default 0.95");
            out.println("  --seed <long>                 random seed, default System.nanoTime()");
            out.println("  --max-tokens, -n <int>        number of steps to run for < 0 = limited by context length, default " + DEFAULT_MAX_TOKENS);
            out.println("  --stream <boolean>            print tokens during generation; accepts true|false|on|off, default true");
            out.println("  --echo <boolean>              print ALL tokens to stderr; accepts true|false|on|off, default false");
            out.println("  --color <on|off|auto>         colorize thinking output in terminal (default: auto)");
            out.println("  --think <off|on|inline>       on: show thinking (default), off: hide thinking from output (model still generates it), inline: thoughts to stdout");
            out.println("  --keep-past-thinking <bool>   keep prior assistant thinking in history (default false)");
            out.println("  --raw-prompt                  bypass chat template and tokenize --prompt directly");
            out.println();
            out.println("Interactive commands:");
            out.println("  /quit, /exit                  exit the chat");
            out.println("  /context                      show context token usage");
            out.println();
            out.println("Examples:");
            out.println("  jbang LFM25.java --model LFM2.5-1.2B-Instruct-Q8_0.gguf --chat");
            out.println("  jbang LFM25.java --model LFM2.5-1.2B-Instruct-Q8_0.gguf --prompt \"Tell me a joke\"");
            out.println("  jbang LFM25.java --model LFM2.5-1.2B-Instruct-Q8_0.gguf --chat --system-prompt \"You are a helpful assistant\"");
        }

        static Options parseOptions(String[] args) {
            String prompt = null;
            String suffix = null;
            String systemPrompt = null;
            float temperature = 1f;
            float topp = 0.95f;
            Path modelPath = null;
            long seed = System.nanoTime();
            int maxTokens = DEFAULT_MAX_TOKENS;
            boolean interactive = false;
            boolean stream = true;
            boolean echo = false;
            boolean think = true;
            boolean thinkInline = false;
            String colorMode = "auto";
            boolean keepPastThinking = false;
            boolean rawPrompt = false;

            for (int i = 0; i < args.length; i++) {
                String optionName = args[i];
                require(optionName.startsWith("-"), "Invalid option %s", optionName);
                switch (optionName) {
                    case "--interactive", "--chat", "-i" -> interactive = true;
                    case "--instruct" -> interactive = false;
                    case "--raw-prompt" -> rawPrompt = true;
                    case "--help", "-h" -> {
                        printUsage(System.out);
                        System.exit(0);
                    }
                    default -> {
                        String nextArg;
                        if (optionName.contains("=")) {
                            String[] parts = optionName.split("=", 2);
                            optionName = parts[0];
                            nextArg = parts[1];
                        } else {
                            require(i + 1 < args.length, "Missing argument for option %s", optionName);
                            nextArg = args[i + 1];
                            i += 1;
                        }
                        switch (optionName) {
                            case "--prompt", "-p" -> prompt = nextArg;
                            case "--suffix" -> suffix = nextArg;
                            case "--system-prompt", "-sp" -> systemPrompt = nextArg;
                            case "--temperature", "--temp" -> temperature = Float.parseFloat(nextArg);
                            case "--top-p" -> topp = Float.parseFloat(nextArg);
                            case "--model", "-m" -> modelPath = Path.of(nextArg);
                            case "--seed", "-s" -> seed = Long.parseLong(nextArg);
                            case "--max-tokens", "-n" -> maxTokens = Integer.parseInt(nextArg);
                            case "--stream" -> stream = parseBooleanOption(optionName, nextArg);
                            case "--echo" -> echo = parseBooleanOption(optionName, nextArg);
                            case "--color" -> colorMode = nextArg.toLowerCase(Locale.ROOT);
                            case "--keep-past-thinking" -> keepPastThinking = parseBooleanOption(optionName, nextArg);
                            case "--think" -> {
                                String thinkMode = nextArg.toLowerCase(Locale.ROOT);
                                thinkInline = List.of("inline", "stdout").contains(thinkMode);
                                switch (thinkMode) {
                                    case "on", "true", "inline", "stdout" -> think = true;
                                    case "off", "false" -> think = false;
                                    default -> require(false, "Invalid argument for %s: expected off|on|inline (or false|true|stdout), got %s", optionName, nextArg);
                                }
                            }
                            default -> require(false, "Unknown option: %s", optionName);
                        }
                    }
                }
            }
            require(List.of("on", "off", "auto").contains(colorMode), "Invalid argument: --color must be one of on|off|auto");
            boolean color = LFM25.supportsAnsiColors(colorMode);
            return new Options(modelPath, prompt, suffix, systemPrompt, interactive, temperature, topp, seed, maxTokens, stream, echo, think, thinkInline, color, keepPastThinking, rawPrompt);
        }
    }

    public static void main(String[] args) throws IOException {
        Options options;
        try {
            options = Options.parseOptions(args);
        } catch (IllegalArgumentException e) {
            System.out.println(e.getMessage());
            System.out.println();
            Options.printUsage(System.out);
            System.exit(-1);
            return;
        }
        Llama model = AOT.tryUsePreLoaded(options.modelPath(), options.maxTokens());
        if (model == null) {
            model = ModelLoader.loadModel(options.modelPath(), options.maxTokens());
        }
        Sampler sampler = selectSampler(model.configuration().vocabularySize, options.temperature(), options.topp(), options.seed());
        if (!options.think()) {
            Integer thinkStart = model.tokenizer().getSpecialTokens().get("<think>");
            Integer thinkEnd = model.tokenizer().getSpecialTokens().get("</think>");
            Set<Integer> banned = new HashSet<>();
            if (thinkStart != null) banned.add(thinkStart);
            if (thinkEnd != null) banned.add(thinkEnd);
            if (!banned.isEmpty()) {
                Sampler inner = sampler;
                sampler = logits -> {
                    for (int token : banned) {
                        logits.setFloat(token, Float.NEGATIVE_INFINITY);
                    }
                    return inner.sampleToken(logits);
                };
            }
        }
        if (options.interactive()) {
            runInteractive(model, sampler, options);
        } else {
            runInstructOnce(model, sampler, options);
        }
    }
}

final class AOT {
    record PartialModel(String modelFileName, Llama model, long tensorDataOffset,
                        Map<String, GGUF.GGUFTensorInfo> tensorInfos,
                        Pair<float[], float[]> ropeFreqsSWA, Pair<float[], float[]> ropeFreqsFull) {}

    private static final PartialModel PRELOADED_GGUF = preLoadGGUF(System.getProperty("lfm2.PreloadGGUF"));

    private static PartialModel preLoadGGUF(String modelPath) {
        if (modelPath == null || modelPath.isEmpty()) {
            return null;
        }
        try {
            Path path = Path.of(modelPath);
            if (!Files.exists(path) || !Files.isRegularFile(path)) {
                throw new IllegalArgumentException("Cannot pre-load model: " + path);
            }
            try (FileChannel fileChannel = FileChannel.open(path, StandardOpenOption.READ)) {
                GGUF gguf = GGUF.loadModel(fileChannel, path.toString());
                Llama base = ModelLoader.loadModel(null, gguf, LFM25.DEFAULT_MAX_TOKENS, false);
                // Precompute RoPE frequencies at build time (pure Java arrays, survives native-image)
                Llama.Configuration config = base.configuration();
                Pair<float[], float[]> ropeFreqsSWA = RoPE.precomputeFreqsCis(
                        config.contextLength, config.headSizeSWA, config.ropeThetaSWA);
                Pair<float[], float[]> ropeFreqsFull;
                Map<String, GGMLTensorEntry> tmpEntries = GGUF.loadTensors(fileChannel, gguf.getTensorDataOffset(), gguf.getTensorInfos());
                GGMLTensorEntry ropeFreqEntry = tmpEntries.get("rope_freqs.weight");
                if (ropeFreqEntry != null) {
                    FloatBuffer ropeFreqsBuf = ModelLoader.toFloatBuffer(ropeFreqEntry);
                    float[] modelRopeFreqs = new float[ropeFreqsBuf.remaining()];
                    ropeFreqsBuf.get(modelRopeFreqs);
                    ropeFreqsFull = RoPE.precomputeFreqsCisFromFreqs(
                            config.contextLength, config.headSizeFull, config.ropeTheta, modelRopeFreqs);
                } else {
                    ropeFreqsFull = RoPE.precomputeFreqsCis(
                            config.contextLength, config.headSizeFull, config.ropeTheta);
                }
                return new PartialModel(
                        path.getFileName().toString(), base,
                        gguf.getTensorDataOffset(), gguf.getTensorInfos(),
                        ropeFreqsSWA, ropeFreqsFull);
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static Llama tryUsePreLoaded(Path modelPath, int contextLength) throws IOException {
        AOT.PartialModel preLoaded = AOT.PRELOADED_GGUF;
        if (preLoaded == null) {
            return null;
        }
        String optionsModel = modelPath.getFileName().toString();
        String preLoadedModel = preLoaded.modelFileName();
        if (!Objects.equals(optionsModel, preLoadedModel)) {
            return null;
        }
        Llama baseModel = preLoaded.model();
        try (var timer = Timer.log("Load tensors from pre-loaded model");
             var fileChannel = FileChannel.open(modelPath, StandardOpenOption.READ)) {
            Map<String, GGMLTensorEntry> tensorEntries = GGUF.loadTensors(fileChannel, preLoaded.tensorDataOffset(), preLoaded.tensorInfos());
            Llama.Weights weights = ModelLoader.loadWeightsWithRoPE(tensorEntries, baseModel.configuration(),
                    preLoaded.ropeFreqsSWA(), preLoaded.ropeFreqsFull());
            return new Llama(baseModel.configuration().withContextLength(contextLength), baseModel.tokenizer(), weights);
        }
    }
}
