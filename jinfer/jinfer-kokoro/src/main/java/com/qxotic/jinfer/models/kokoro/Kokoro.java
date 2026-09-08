package com.qxotic.jinfer.models.kokoro;

import com.qxotic.format.gguf.GGMLType;
import com.qxotic.format.gguf.GGUF;
import com.qxotic.format.gguf.TensorEntry;
import com.qxotic.jinfer.Arenas;
import com.qxotic.jinfer.LeakWatch;
import com.qxotic.jinfer.RuntimeState;
import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.kernels.Convert;
import com.qxotic.jinfer.kernels.ModelLoader;
import com.qxotic.jota.memory.MemoryAllocator;
import com.qxotic.jota.memory.MemoryAllocators;
import com.qxotic.jota.memory.MemoryArena;
import com.qxotic.jota.memory.MemoryView;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.ref.Reference;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.Map;
import java.util.Random;

/** Kokoro v1.0 weights and one length-indexed voice pack. */
public final class Kokoro {

    public static final String ARCHITECTURE = "kokoro";
    public static final String VOICE_ARCHITECTURE = "kokoro-voice";
    public static final int MAX_PHONEMES = 510;

    /** Dimensions carried by the cstr/simonfxr Kokoro GGUF schema. */
    public record Configuration(
            int dimIn,
            int hiddenDim,
            int styleDim,
            int maxConvDim,
            int maxDuration,
            int tokenCount,
            int melCount,
            int decoderLayers,
            int sampleRate,
            int textEncoderKernelSize,
            int istftInitialChannels,
            int istftFftSize,
            int istftHopSize,
            int[] upsampleRates,
            int[] upsampleKernelSizes,
            int[] resblockKernelSizes,
            int[] resblockDilations,
            int plbertEmbeddingSize,
            int plbertHiddenSize,
            int plbertLayers,
            int plbertHeads,
            int plbertIntermediateSize,
            int plbertMaxPositions,
            String[] tokens) {

        public Configuration {
            upsampleRates = upsampleRates.clone();
            upsampleKernelSizes = upsampleKernelSizes.clone();
            resblockKernelSizes = resblockKernelSizes.clone();
            resblockDilations = resblockDilations.clone();
            tokens = tokens.clone();
        }

        @Override
        public int[] upsampleRates() {
            return upsampleRates.clone();
        }

        @Override
        public int[] upsampleKernelSizes() {
            return upsampleKernelSizes.clone();
        }

        @Override
        public int[] resblockKernelSizes() {
            return resblockKernelSizes.clone();
        }

        @Override
        public int[] resblockDilations() {
            return resblockDilations.clone();
        }

        @Override
        public String[] tokens() {
            return tokens.clone();
        }
    }

    /** A voice row contains decoder style first, then predictor style. */
    public record Voice(int maxPhonemes, int styleDimensions) {}

    /** File-backed tensors; their lifetime is owned by the arena passed to {@link #load}. */
    public record Weights(
            Map<String, MemoryView<MemorySegment>> model, MemoryView<MemorySegment> voicePack) {
        public Weights {
            model = Map.copyOf(model);
        }
    }

    private final Configuration configuration;
    private final Voice voice;
    private final String language;
    private final Weights weights;
    private final TextEncoder.Weights textEncoder;
    private final PlBert.Weights plBert;
    private final ProsodyPredictor.Weights predictor;
    private final KokoroDecoder.Weights decoder;
    private final KokoroGenerator.Weights generator;
    private final float[] sourceWeight;
    private final float sourceBias;
    private final Symbols symbols;
    private final long parameterCount;

    private Kokoro(
            Configuration configuration,
            Voice voice,
            String language,
            Weights weights,
            TextEncoder.Weights textEncoder,
            PlBert.Weights plBert,
            ProsodyPredictor.Weights predictor,
            KokoroDecoder.Weights decoder,
            KokoroGenerator.Weights generator,
            float[] sourceWeight,
            float sourceBias,
            long parameterCount) {
        this.configuration = configuration;
        this.voice = voice;
        this.language = language;
        this.weights = weights;
        this.textEncoder = textEncoder;
        this.plBert = plBert;
        this.predictor = predictor;
        this.decoder = decoder;
        this.generator = generator;
        this.sourceWeight = sourceWeight;
        this.sourceBias = sourceBias;
        this.symbols = new Symbols(configuration.tokens);
        this.parameterCount = parameterCount;
    }

    public static Kokoro load(Path model, Path voice, Arena arena) throws IOException {
        try (FileChannel channel = FileChannel.open(model, StandardOpenOption.READ)) {
            return load(channel, ModelLoader.readGguf(channel, model.toString()), voice, arena);
        }
    }

    /** Loads a pre-parsed base model and a separate voice GGUF into the same arena. */
    public static Kokoro load(FileChannel channel, GGUF gguf, Path voice, Arena arena)
            throws IOException {
        return load(channel, gguf, 0, voice, arena);
    }

    /** Loads a pre-parsed base model embedded at {@code baseOffset}. */
    public static Kokoro load(
            FileChannel channel, GGUF gguf, long baseOffset, Path voice, Arena arena)
            throws IOException {
        Configuration configuration = readConfig(gguf);

        try (FileChannel voiceChannel = FileChannel.open(voice, StandardOpenOption.READ)) {
            GGUF voiceGguf = ModelLoader.readGguf(voiceChannel, voice.toString());
            Voice voiceConfiguration = readVoice(voiceGguf);
            String language = readVoiceLanguage(voiceGguf);
            Map<String, MemoryView<MemorySegment>> modelWeights =
                    ModelLoader.loadTensors(channel, gguf, baseOffset, arena);
            MemoryAllocator<MemorySegment> persistent = MemoryAllocators.ofArena(arena);
            TextEncoder.Weights textEncoder =
                    TextEncoder.load(modelWeights, configuration, persistent);
            PlBert.Weights plBert = PlBert.load(modelWeights, configuration, persistent);
            ProsodyPredictor.Weights predictor =
                    ProsodyPredictor.load(modelWeights, configuration, persistent);
            KokoroDecoder.Weights decoder = KokoroDecoder.load(modelWeights, persistent);
            KokoroGenerator.Weights generator =
                    KokoroGenerator.load(modelWeights, configuration, persistent);
            MemoryView<MemorySegment> sourceWeightView =
                    ModelLoader.require(modelWeights, "dec.gen.m_source.weight");
            require(sourceWeightView.logicalSize() == 9, "m_source weight has the wrong size");
            MemoryView<MemorySegment> sourceWeightF32 = Views.allocateF32(persistent, 9);
            Convert.copyToF32(sourceWeightView, 0, sourceWeightF32, 0, 9);
            float[] sourceWeight = Views.toFloatArray(sourceWeightF32, "m_source weight");
            MemoryView<MemorySegment> sourceBias =
                    KokoroLayers.vector(modelWeights, persistent, "dec.gen.m_source.bias", 1);
            MemoryView<MemorySegment> voicePack =
                    ModelLoader.requireF32(
                            ModelLoader.loadTensors(voiceChannel, voiceGguf, arena), "voice.pack");
            long parameters =
                    gguf.getTensors().stream().mapToLong(TensorEntry::totalNumberOfElements).sum();
            return new Kokoro(
                    configuration,
                    voiceConfiguration,
                    language,
                    new Weights(modelWeights, voicePack),
                    textEncoder,
                    plBert,
                    predictor,
                    decoder,
                    generator,
                    sourceWeight,
                    Views.getFloat(sourceBias, 0, "m_source bias"),
                    parameters);
        }
    }

    static Configuration readConfig(GGUF gguf) {
        require(
                ARCHITECTURE.equals(gguf.getString("general.architecture")),
                "unsupported architecture '" + gguf.getString("general.architecture") + "'");

        Configuration config =
                new Configuration(
                        integer(gguf, "kokoro.dim_in"),
                        integer(gguf, "kokoro.hidden_dim"),
                        integer(gguf, "kokoro.style_dim"),
                        integer(gguf, "kokoro.max_conv_dim"),
                        integer(gguf, "kokoro.max_dur"),
                        integer(gguf, "kokoro.n_token"),
                        integer(gguf, "kokoro.n_mels"),
                        integer(gguf, "kokoro.n_layer"),
                        integer(gguf, "kokoro.sample_rate"),
                        integer(gguf, "kokoro.text_encoder_kernel_size"),
                        integer(gguf, "kokoro.istft.init_channel"),
                        integer(gguf, "kokoro.istft.n_fft"),
                        integer(gguf, "kokoro.istft.hop_size"),
                        integers(gguf, "kokoro.istft.upsample_rates"),
                        integers(gguf, "kokoro.istft.upsample_kernel_sizes"),
                        integers(gguf, "kokoro.istft.resblock_kernel_sizes"),
                        integers(gguf, "kokoro.istft.resblock_dilation_sizes"),
                        integer(gguf, "kokoro.plbert.embedding_size"),
                        integer(gguf, "kokoro.plbert.hidden_size"),
                        integer(gguf, "kokoro.plbert.num_hidden_layers"),
                        integer(gguf, "kokoro.plbert.num_attention_heads"),
                        integer(gguf, "kokoro.plbert.intermediate_size"),
                        integer(gguf, "kokoro.plbert.max_position_embeddings"),
                        strings(gguf, "tokenizer.ggml.tokens"));

        require(
                config.dimIn == 64
                        && config.hiddenDim == 512
                        && config.styleDim == 128
                        && config.maxConvDim == 512
                        && config.maxDuration == 50,
                "unsupported model dimensions");
        require(
                config.tokenCount == 178 && config.tokens.length == config.tokenCount,
                "symbol table must contain 178 entries");
        require(
                config.sampleRate == 24_000
                        && config.melCount == 80
                        && config.decoderLayers == 3
                        && config.textEncoderKernelSize == 5
                        && config.istftInitialChannels == 512
                        && config.istftFftSize == 20
                        && config.istftHopSize == 5,
                "unsupported audio layout");
        require(
                Arrays.equals(config.upsampleRates, new int[] {10, 6})
                        && Arrays.equals(config.upsampleKernelSizes, new int[] {20, 12})
                        && Arrays.equals(config.resblockKernelSizes, new int[] {3, 7, 11})
                        && Arrays.equals(
                                config.resblockDilations, new int[] {1, 3, 5, 1, 3, 5, 1, 3, 5}),
                "unsupported ISTFTNet layout");
        require(
                config.plbertEmbeddingSize == 128
                        && config.plbertHiddenSize == 768
                        && config.plbertLayers == 12
                        && config.plbertHeads == 12
                        && config.plbertIntermediateSize == 2048
                        && config.plbertMaxPositions == 512,
                "unsupported PL-BERT layout");
        return config;
    }

    static Voice readVoice(GGUF gguf) {
        require(
                VOICE_ARCHITECTURE.equals(gguf.getString("general.architecture")),
                "unsupported voice architecture '" + gguf.getString("general.architecture") + "'");
        TensorEntry pack = gguf.getTensor("voice.pack");
        require(pack != null, "missing tensor: voice.pack");
        require(pack.ggmlType() == GGMLType.F32, "voice.pack must be F32");
        long[] shape = pack.shape();
        require(
                shape.length == 3 && shape[0] == 256 && shape[1] == 1 && shape[2] >= MAX_PHONEMES,
                "voice.pack must have GGUF shape [256, 1, max_phonemes]");
        require(shape[2] <= Integer.MAX_VALUE, "voice pack is too large");
        return new Voice(Math.toIntExact(shape[2]), Math.toIntExact(shape[0]));
    }

    static String readVoiceLanguage(GGUF gguf) {
        require(gguf.containsKey("kokoro_voice.name"), "voice is missing kokoro_voice.name");
        String name = gguf.getString("kokoro_voice.name");
        require(!name.isEmpty(), "voice name is empty");
        return switch (name.charAt(0)) {
            case 'a' -> "en-us";
            case 'b' -> "en-gb";
            case 'e' -> "es";
            case 'f' -> "fr-fr";
            case 'h' -> "hi";
            case 'i' -> "it";
            case 'j' -> "ja";
            case 'p' -> "pt-br";
            case 'z' -> "cmn";
            default ->
                    throw new IllegalArgumentException(
                            "Kokoro: unsupported voice language: " + name);
        };
    }

    private static int integer(GGUF gguf, String key) {
        require(gguf.containsKey(key), "missing metadata: " + key);
        return gguf.getValue(int.class, key);
    }

    private static int[] integers(GGUF gguf, String key) {
        require(gguf.containsKey(key), "missing metadata: " + key);
        return gguf.getValue(int[].class, key);
    }

    private static String[] strings(GGUF gguf, String key) {
        require(gguf.containsKey(key), "missing metadata: " + key);
        return gguf.getValue(String[].class, key);
    }

    private static void require(boolean condition, String message) {
        if (!condition) throw new IllegalArgumentException("Kokoro: " + message);
    }

    public Configuration configuration() {
        return configuration;
    }

    public Voice voice() {
        return voice;
    }

    String language() {
        return language;
    }

    public Weights weights() {
        return weights;
    }

    public Symbols symbols() {
        return symbols;
    }

    public long parameterCount() {
        return parameterCount;
    }

    private MemoryView<MemorySegment> style(
            int rawPhonemes, int offset, MemoryAllocator<MemorySegment> allocator) {
        int row = Math.min(rawPhonemes, voice.maxPhonemes) - 1;
        int width = configuration.styleDim;
        MemoryView<MemorySegment> result = Views.allocateF32(allocator, 1, width);
        Convert.copyF32(
                weights.voicePack, (long) row * voice.styleDimensions + offset, result, 0, width);
        return result;
    }

    public State newState() {
        MemoryArena<MemorySegment> arena = Arenas.newCrossThreadMemoryArena();
        try {
            return new State(arena, arena);
        } catch (RuntimeException | Error e) {
            Arenas.close(arena);
            throw e;
        }
    }

    public State newState(MemoryArena<MemorySegment> arena) {
        if (arena == null) throw new IllegalArgumentException("null arena");
        return new State(arena, null);
    }

    /** Synthesizes one non-empty sequence of raw Kokoro phoneme IDs. */
    public float[] synthesize(State state, int[] rawPhonemes, double speed, long seed) {
        float[] result =
                state.exclusively(
                        () -> {
                            require(rawPhonemes.length > 0, "phoneme input is empty");
                            require(rawPhonemes.length <= MAX_PHONEMES, "too many phonemes");
                            require(
                                    Double.isFinite(speed) && speed >= 0.5 && speed <= 2,
                                    "speed must be in [0.5, 2.0]");
                            Views.checkAlive(weights.voicePack, "voice pack");
                            state.scratch.rewind();
                            int[] tokens = state.scratch.ints(rawPhonemes.length + 2);
                            Arrays.fill(tokens, 0);
                            System.arraycopy(rawPhonemes, 0, tokens, 1, rawPhonemes.length);
                            MemoryAllocator<MemorySegment> scratch = state.scratch;
                            MemoryView<MemorySegment> predictorStyle =
                                    style(rawPhonemes.length, configuration.styleDim, scratch);
                            MemoryView<MemorySegment> decoderStyle =
                                    style(rawPhonemes.length, 0, scratch);
                            ProsodyPredictor.Output prediction =
                                    ProsodyPredictor.forward(
                                            predictor,
                                            PlBert.encode(plBert, tokens, configuration, scratch),
                                            predictorStyle,
                                            tokens.length,
                                            speed,
                                            scratch);
                            MemoryView<MemorySegment> decoded =
                                    KokoroDecoder.forward(
                                            decoder,
                                            TextEncoder.encode(
                                                    textEncoder,
                                                    tokens,
                                                    configuration.hiddenDim,
                                                    configuration.textEncoderKernelSize,
                                                    scratch),
                                            prediction.alignmentIndices(),
                                            prediction.f0(),
                                            prediction.noise(),
                                            decoderStyle,
                                            scratch);
                            KokoroDsp.Spectrum harmonic =
                                    KokoroDsp.sourceStft(
                                            prediction.f0(),
                                            sourceWeight,
                                            sourceBias,
                                            state.random(seed),
                                            scratch);
                            KokoroDsp.Spectrum generated =
                                    KokoroGenerator.forward(
                                            generator, decoded, harmonic, decoderStyle, scratch);
                            float[] pcm =
                                    KokoroDsp.istft(
                                            generated.magnitude(), generated.phase(), scratch);
                            for (int i = 0; i < pcm.length; i++)
                                pcm[i] = Math.clamp(pcm[i], -1f, 1f);
                            return pcm;
                        });
        Reference.reachabilityFence(this);
        return result;
    }

    public static final class State extends RuntimeState {
        private final MemoryArena<MemorySegment> owned;
        private final MemoryArena<MemorySegment> allocator;
        private final KokoroWorkspace scratch;
        private final Runnable disarm;
        private final Random random = new Random();

        private State(MemoryArena<MemorySegment> allocator, MemoryArena<MemorySegment> owned) {
            this.owned = owned;
            this.allocator = allocator;
            this.scratch = new KokoroWorkspace(allocator);
            this.disarm = LeakWatch.arm(this, "Kokoro.State");
        }

        private Random random(long seed) {
            random.setSeed(seed);
            return random;
        }

        int scratchAllocations() {
            return scratch.backingAllocations();
        }

        @Override
        protected void checkResourcesAlive() {
            if (!allocator.isAlive())
                throw new IllegalStateException("the speech state's arena has been closed");
        }

        @Override
        protected void releaseResources() {
            disarm.run();
            if (owned != null) Arenas.close(owned);
        }
    }
}
