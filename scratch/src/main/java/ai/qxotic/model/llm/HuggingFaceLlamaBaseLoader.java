package ai.qxotic.model.llm;

import ai.qxotic.format.gguf.GGUF;
import ai.qxotic.format.safetensors.DType;
import ai.qxotic.format.safetensors.Safetensors;
import ai.qxotic.format.safetensors.SafetensorsIndex;
import ai.qxotic.format.safetensors.TensorEntry;
import ai.qxotic.model.llm.llama.*;
import ai.qxotic.span.FloatMatrixView;
import ai.qxotic.span.FloatSpan;
import ai.qxotic.span.FloatUnaryOperator;
import ai.qxotic.tokenizers.Normalizer;
import ai.qxotic.tokenizers.TextSplitter;
import ai.qxotic.tokenizers.Tokenizer;
import ai.qxotic.tokenizers.impl.RegexSplitter;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Collection;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;
import java.util.function.IntFunction;
import java.util.stream.Collectors;

public class HuggingFaceLlamaBaseLoader
        extends AbstractHuggingFaceLoader<Llama, Llama.Configuration, Llama.Weights, Llama.State> {
    protected HuggingFaceLlamaBaseLoader(SafetensorsIndex safetensorsIndex) {
        super(safetensorsIndex);
    }

    @Override
    public Llama loadModel(Llama.Configuration configuration) {
        return null;
    }

    @Override
    public ChatFormat createChatFormat(Tokenizer tokenizer) {
        return null;
    }

    @Override
    public Llama.Configuration loadConfiguration(int maxTokens, SpanLoader spanLoader) {
        return null;
    }

    @Override
    public Llama.Weights loadWeights(Llama.Configuration configuration, SpanLoader spanLoader) {
        return null;
    }

//    protected HuggingFaceLlamaBaseLoader(SafetensorsIndex safetensorsIndex) {
//        super(safetensorsIndex);
//    }
//
//    static Map<String, TensorEntry> loadTensorEntries(Path modelPath) throws IOException {
//        if (Files.isDirectory(modelPath)) {
//            throw new IllegalArgumentException("expecte a file path");
//        }
//        Collection<TensorEntry> tensors = Safetensors.read(modelPath).getTensors();
//        return tensors.stream().collect(Collectors.toMap(TensorEntry::name, Function.identity()));
//    }
//
//    public static void main(String[] args) throws IOException {
//        Path modelRoot = Path.of("/home/mukel/Desktop/playground/models/hf/Llama-3.2-1B-Instruct");
//        Options options = Options.parseOptions(args);
//
//        SafetensorsIndex index = SafetensorsIndex.load(modelRoot);
//        var loader = new HuggingFaceLlamaBaseLoader(index);
//
//        Llama.Weights weights;
//        Llama.Configuration configuration;
//        try (var fileChannel =
//                FileChannel.open(modelRoot.resolve("model.safetensors"), StandardOpenOption.READ)) {
//            SpanLoader spanLoader = loaderFromTensorDataMM(0, fileChannel);
//            configuration = loader.loadConfiguration(options.maxTokens(), spanLoader);
//            weights = loader.loadWeights(configuration, spanLoader);
//        }
//
//        TextSplitter textSplitter =
//                RegexSplitter.create(LlamaBPETextSplitterFactory.LLAMA3_PATTERN);
//        GGUF gguf =
//                GGUF.read(
//                        Path.of(
//                                "/home/mukel/Desktop/playground/models/hf/Llama-3.2-1B-Instruct/Llama-3.2-1B-Instruct-BF16.gguf"));
//        Tokenizer tokenizer =
//                new TokenizerLoader().loadTokenizer(gguf, Normalizer.IDENTITY, textSplitter);
//
//        Sampler2 sampler =
//                Sampler2.selectSampler(
//                        tokenizer.vocabulary().size(),
//                        options.temperature(),
//                        options.topp(),
//                        options.seed());
//
//        ChatFormat chatFormat = new Llama3ChatFormat(tokenizer);
//        Llama model = loader.loadModel(configuration);
//
//        RunInteractive.runInteractive(
//                model, weights, chatFormat, sampler.fromLogits(s -> s.logits), options);
//    }
//
//    //    private static Vocabulary loadVocabulary(Map<String, Object> jsonConfig, Path
//    // tokenizerBinPath) throws IOException {
//    //        if (!Files.exists(tokenizerBinPath)) {
//    //            throw new IllegalArgumentException(tokenizerBinPath + " not found, please convert
//    // HF tokenizer.model into tokenizer.bin using Tokenizer.java jbang script");
//    //        }
//    //        Vocabulary binVocabulary =
//    // LLama2CLoader.loadVocabulary(JSON.asNumber(jsonConfig.get("vocab_sie")).intValue(),
//    // tokenizerBinPath);
//    //        int bosTokenId = JSON.asNumber(jsonConfig.get("bos_token_id")).intValue();
//    //        int eosTokenId = JSON.asNumber(jsonConfig.get("eos_token_id")).intValue();
//    //        return new Vocabulary(binVocabulary.vocabulary(), binVocabulary.scores(),
//    // OptionalInt.of(bosTokenId), OptionalInt.of(eosTokenId), OptionalInt.empty(),
//    // OptionalInt.empty());
//    //    }
//
//    //
//    //    static FloatBuffer[] loadArrayOfFloatBuffers(int size, IntFunction<TensorEntry>
//    // getTensorEntry) {
//    //        FloatBuffer[] array = new FloatBuffer[size];
//    //        for (int i = 0; i < size; i++) {
//    //            array[i] = Conversions.toF32(getTensorEntry.apply(i));
//    //        }
//    //        return array;
//    //    }
//    //
//    //    static FloatBuffer[] loadArrayOfFloatBuffersRearrangeHuggingFace(Llama2.Config config, int
//    // size, IntFunction<TensorEntry> getTensorEntry) {
//    //        ByteBuffer[] byteBuffers = loadArrayOfByteBuffersRearrangeHuggingFace(config, size,
//    // getTensorEntry);
//    //        FloatBuffer[] floatBuffers = new FloatBuffer[byteBuffers.length];
//    //        for (int i = 0; i < floatBuffers.length; ++i) {
//    //            floatBuffers[i] = byteBuffers[i].asFloatBuffer();
//    //        }
//    //        return floatBuffers;
//    //    }
//
//    //    static ByteBuffer[] loadArrayOfByteBuffersRearrangeHuggingFace(Llama2.Config config, int
//    // size, IntFunction<TensorEntry> getTensorEntry) {
//    //        ByteBuffer[] array = new ByteBuffer[size];
//    //        for (int i = 0; i < size; i++) {
//    //            TensorEntry tensorEntry = getTensorEntry.apply(i);
//    //            int n = tensorEntry.numberOfElements();
//    //            ByteBuffer bb = ByteBuffer.allocate(n *
//    // Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
//    //            FloatBuffer meta = bb.asFloatBuffer();
//    //            int headPiece = config.headSize / 2;
//    //            int elementsPerHead = n / config.numberOfHeads;
//    //            for (int h = 0; h < config.numberOfHeads; h++) {
//    //                int hOffset = h * elementsPerHead;
//    //                for (int chunk = 0; chunk < elementsPerHead / config.dim; ++chunk) {
//    //                    // Undo the following permutation:
//    //                    //
//    // https://github.com/huggingface/transformers/blob/eec5841e9f440c795fb9292d009675d97a14f983/src/transformers/models/llama/convert_llama_weights_to_hf.py#L113-L115
//    //                    // # permute for sliced rotary
//    //                    // def permute(w, n_heads=n_heads, dim1=dim, dim2=dim):
//    //                    //     return w.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1,
//    // 2).reshape(dim1, dim2)
//    //                    int srcOffset;
//    //                    if (chunk % 2 == 0) {
//    //                        srcOffset = hOffset + chunk / 2 * config.dim;
//    //                    } else {
//    //                        srcOffset = hOffset + (chunk / 2 + headPiece) * config.dim;
//    //                    }
//    //                    Conversions.toF32(tensorEntry, srcOffset, meta, hOffset + chunk *
//    // config.dim, config.dim);
//    //                }
//    //            }
//    //            array[i] = bb;
//    //        }
//    //        return array;
//    //    }
//    //
//    //    static FloatTensor[] loadArrayOfFloatsRearrangeHuggingFace(Llama2.Config config, int size,
//    // IntFunction<TensorEntry> getTensorEntry) {
//    //        ByteBuffer[] byteBuffers = loadArrayOfByteBuffersRearrangeHuggingFace(config, size,
//    // getTensorEntry);
//    //        FloatTensor[] floats = new FloatTensor[byteBuffers.length];
//    //        for (int i = 0; i < floats.length; ++i) {
//    //            floats[i] = new F32FloatTensor(size, MemorySegment.ofBuffer(byteBuffers[i]));
//    //        }
//    //        return floats;
//    //    }
//    //
//    //    static FloatTensor[] loadArrayOfFloats(int size, IntFunction<TensorEntry>
//    // getTensorEntry) {
//    //        FloatTensor[] array = new FloatTensor[size];
//    //        for (int i = 0; i < size; i++) {
//    //            array[i] = HFConversions.toFloats(getTensorEntry.apply(i));
//    //        }
//    //        return array;
//    //    }
//
//    public static SpanLoader loaderFromTensorDataMM(
//            long tensorDataOffset, FileChannel fileChannel) {
//        return new SpanLoader() {
//            @Override
//            public void close() throws Exception {}
//
//            @Override
//            public FloatSpan apply(Object baseTensor) {
//                TensorEntry tensorEntry = (TensorEntry) baseTensor;
//                DType baseType = tensorEntry.dtype();
//                long sizeInBytes = baseType.byteSizeForShape(tensorEntry.shape());
//                MemorySegment memorySegment = null;
//                try {
//                    memorySegment =
//                            fileChannel.map(
//                                    FileChannel.MapMode.READ_ONLY,
//                                    tensorDataOffset + tensorEntry.byteOffset(),
//                                    sizeInBytes,
//                                    Arena.ofAuto());
//                } catch (IOException e) {
//                    throw new UncheckedIOException(e);
//                }
//                FloatSpan span =
//                        switch (baseType) {
//                            case F32 -> new F32Span(memorySegment);
//                            //                case Q4_0 -> new Q4_0Span(memorySegment);
//                            //                case Q8_0 -> new Q8_0Span(memorySegment);
//                            //                case Q4_1 -> new Q4_1Span(memorySegment);
//                            case BF16 -> new BF16Span(memorySegment);
//                            default ->
//                                    throw new UnsupportedOperationException(
//                                            "Quantization not supported " + baseType);
//                        };
//                assert span.size() == Util.numberOfElements(tensorEntry.shape());
//                return span;
//            }
//        };
//    }
//
//    @Override
//    public Llama loadModel(Llama.Configuration configuration) {
//        return new Llama(
//                configuration, DefaultKernelOps.getKernelOps(), DefaultKernelOps.getSpanFactory());
//    }
//
//    public ChatFormat createChatFormat(Tokenizer tokenizer) {
//        return new Llama3ChatFormat(tokenizer);
//    }
//
//    @Override
//    public Llama.Configuration loadConfiguration(int maxTokens, SpanLoader spanLoader) {
//        Map<String, Object> modelConfig = safetensorsIndex.loadModelConfig();
//        int embeddingLength = ((Number) modelConfig.get("hidden_size")).intValue();
//        int numberOfHeads = ((Number) modelConfig.get("num_attention_heads")).intValue();
//        return new Llama.Configuration(
//                embeddingLength,
//                ((Number) modelConfig.get("intermediate_size")).intValue(),
//                ((Number) modelConfig.get("num_hidden_layers")).intValue(),
//                numberOfHeads,
//                embeddingLength / numberOfHeads,
//                embeddingLength / numberOfHeads,
//                ((Number) modelConfig.get("num_key_value_heads")).intValue(),
//                ((Number) modelConfig.get("vocab_size")).intValue(),
//                ((Number) modelConfig.get("max_position_embeddings")).intValue(),
//                ((Number) modelConfig.get("rms_norm_eps")).floatValue(),
//                ((Number) modelConfig.get("rope_theta")).floatValue(),
//                true,
//                ((Number) modelConfig.getOrDefault("attention_multiplier", Float.NaN)).floatValue(),
//                ((Number) modelConfig.getOrDefault("residual_multiplier", Float.NaN)).floatValue(),
//                ((Number) modelConfig.getOrDefault("logits_scaling", Float.NaN)).floatValue(),
//                ((Number) modelConfig.getOrDefault("embedding_multiplier", Float.NaN)).floatValue(),
//                FloatUnaryOperator.SILU,
//                null);
//    }
//
//    static Pair<Path, TensorEntry> getTensorEntry(Map<Path, Safetensors> cache, String tensorName) {
//        cache.computeIfAbsent(safetensorsIndex.getSafetensorsPath("model.embed_tokens.weight"), path -> Safetensors.read(path))
//    }
//
//    @Override
//    public Llama.Weights loadWeights(Llama.Configuration config, SpanLoader spanLoader) {
//
//        Map<Path, Safetensors> cache = new ConcurrentHashMap<>();
//
//        try (var timer = Timer.log("Load weights")) {
//            float[][] ropeFreqs =
//                    RoPE.precomputeFreqsCis(
//                            config.contextLength, config.headSize, config.ropeTheta);
//            assert ropeFreqs.length == 2;
//            float[] ropeFreqsReal = ropeFreqs[0];
//            float[] ropeFreqsImag = ropeFreqs[1];
//
//            FloatMatrixView tokenEmbeddings =
//                    tensorAsMatrix(
//
//                            , spanLoader, true);
//
//            // If "output.weight" is not present then the embedding weights are tied/shared with the
//            // decoder.
//            // This is commonly referred as "tie word embeddings".
//            FloatMatrixView classifierWeights =
//                    safetensorsIndex.containsTensor("lm_head.weight")
//                            ? tensorAsMatrix(
//                                    safetensorsIndex.getTensor("lm_head.weight"), spanLoader, true)
//                            : tokenEmbeddings;
//
//            return new Llama.Weights(
//                    tokenEmbeddings,
//                    loadSpanArray(
//                            config.numberOfLayers,
//                            i ->
//                                    safetensorsIndex.getTensor(
//                                            "model.layers." + i + ".input_layernorm.weight"),
//                            spanLoader),
//                    loadMatrixArray(
//                            config.numberOfLayers,
//                            i ->
//                                    safetensorsIndex.getTensor(
//                                            "model.layers." + i + ".self_attn.q_proj.weight"),
//                            spanLoader),
//                    loadMatrixArray(
//                            config.numberOfLayers,
//                            i ->
//                                    safetensorsIndex.getTensor(
//                                            "model.layers." + i + ".self_attn.k_proj.weight"),
//                            spanLoader),
//                    loadMatrixArray(
//                            config.numberOfLayers,
//                            i ->
//                                    safetensorsIndex.getTensor(
//                                            "model.layers." + i + ".self_attn.v_proj.weight"),
//                            spanLoader),
//
//                    // TODO(peterssen): Find correct HF tensor names for Qwen 3.
//                    // qk norm (Qwen 3)
//                    loadSpanArray(
//                            true,
//                            config.numberOfLayers,
//                            i -> safetensorsIndex.getTensor("blk." + i + ".attn_q_norm.weight"),
//                            spanLoader),
//                    loadSpanArray(
//                            true,
//                            config.numberOfLayers,
//                            i -> safetensorsIndex.getTensor("blk." + i + ".attn_k_norm.weight"),
//                            spanLoader),
//
//                    // qkv bias can be null.
//                    loadSpanArray(
//                            true,
//                            config.numberOfLayers,
//                            i ->
//                                    safetensorsIndex.getTensor(
//                                            "model.layers." + i + ".self_attn.q_proj.bias"),
//                            spanLoader),
//                    loadSpanArray(
//                            true,
//                            config.numberOfLayers,
//                            i ->
//                                    safetensorsIndex.getTensor(
//                                            "model.layers." + i + ".self_attn.k_proj.bias"),
//                            spanLoader),
//                    loadSpanArray(
//                            true,
//                            config.numberOfLayers,
//                            i ->
//                                    safetensorsIndex.getTensor(
//                                            "model.layers." + i + ".self_attn.v_proj.bias"),
//                            spanLoader),
//                    loadMatrixArray(
//                            config.numberOfLayers,
//                            i ->
//                                    safetensorsIndex.getTensor(
//                                            "model.layers." + i + ".self_attn.o_proj.weight"),
//                            spanLoader),
//                    loadSpanArray(
//                            config.numberOfLayers,
//                            i ->
//                                    safetensorsIndex.getTensor(
//                                            "model.layers."
//                                                    + i
//                                                    + ".post_attention_layernorm.weight"),
//                            spanLoader),
//                    loadMatrixArray(
//                            config.numberOfLayers,
//                            i ->
//                                    safetensorsIndex.getTensor(
//                                            "model.layers." + i + ".mlp.gate_proj.weight"),
//                            spanLoader),
//                    loadMatrixArray(
//                            config.numberOfLayers,
//                            i ->
//                                    safetensorsIndex.getTensor(
//                                            "model.layers." + i + ".mlp.down_proj.weight"),
//                            spanLoader),
//                    loadMatrixArray(
//                            config.numberOfLayers,
//                            i -> safetensorsIndex.getTensor("model.layers." + i + ".mlp.up_proj.weight"),
//                            spanLoader),
//                    spanLoader.apply(safetensorsIndex.getTensor("model.norm.weight")),
//                    ArraySpan.wrap(ropeFreqsReal),
//                    ArraySpan.wrap(ropeFreqsImag),
//                    classifierWeights);
//        }
//    }
//
//    public static FloatMatrixView tensorAsMatrix(
//            TensorEntry tensorInfo, SpanLoader spanLoader, boolean precomputeRowSpans) {
//        long rows = tensorInfo.shape()[0];
//        long cols = tensorInfo.shape()[1];
//        if (tensorInfo.shape().length != 2) {
//            throw new IllegalArgumentException();
//        }
//        FloatSpan span = spanLoader.apply(tensorInfo);
//        if (rows * cols != span.size()) {
//            throw new IllegalArgumentException();
//        }
//        return FloatMatrixView.asMatrix(span, 0, rows, cols, cols, precomputeRowSpans);
//    }
//
//    public static FloatMatrixView tensorAsMatrix(TensorEntry tensorInfo, SpanLoader spanLoader) {
//        return tensorAsMatrix(tensorInfo, spanLoader, false);
//    }
//
//    public static FloatSpan[] loadSpanArray(
//            int size, IntFunction<TensorEntry> getTensorEntry, SpanLoader spanLoader) {
//        return loadSpanArray(false, size, getTensorEntry, spanLoader);
//    }
//
//    public static FloatMatrixView[] loadMatrixArray(
//            int size, IntFunction<TensorEntry> getTensorEntry, SpanLoader spanLoader) {
//        // return IntStream.range(0,
//        // size).mapToObj(getTensorEntry).map(spanLoader).toArray(FloatSpan[]::new);
//        FloatMatrixView[] array =
//                null; // Lazily allocated if there's at it contains at least one entry.
//        for (int i = 0; i < size; i++) {
//            TensorEntry tensorInfo = getTensorEntry.apply(i);
//            if (tensorInfo != null) {
//                if (array == null) {
//                    array = new FloatMatrixView[size];
//                }
//                array[i] = tensorAsMatrix(tensorInfo, spanLoader);
//            }
//        }
//        if (array == null) {
//            throw new IllegalArgumentException("non-optional tensor wasn't found");
//        }
//        return array;
//    }
//
//    public static FloatSpan[] loadSpanArray(
//            boolean isOptional,
//            int size,
//            IntFunction<TensorEntry> getTensorEntry,
//            SpanLoader spanLoader) {
//        // return IntStream.range(0,
//        // size).mapToObj(getTensorEntry).map(spanLoader).toArray(FloatSpan[]::new);
//        FloatSpan[] array = null; // Lazily allocated if there's at it contains at least one entry.
//        for (int i = 0; i < size; i++) {
//            TensorEntry tensorInfo = getTensorEntry.apply(i);
//            if (tensorInfo != null) {
//                if (array == null) {
//                    array = new FloatSpan[size];
//                }
//                // array[i] = spanLoader.apply(tensorInfo);
//                array[i] = spanLoader.apply(tensorInfo);
//            }
//        }
//        if (array == null && !isOptional) {
//            throw new IllegalArgumentException("non-optional tensor wasn't found");
//        }
//        return array;
//    }
}
