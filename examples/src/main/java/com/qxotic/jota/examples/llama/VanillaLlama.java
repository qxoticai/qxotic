package com.qxotic.jota.examples.llama;

import com.qxotic.format.gguf.GGMLType;
import com.qxotic.format.gguf.GGUF;
import com.qxotic.format.gguf.TensorEntry;
import com.qxotic.jota.DataType;
import com.qxotic.jota.Environment;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryAccess;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.memory.impl.MemoryFactory;
import com.qxotic.jota.tensor.Tensor;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.gguf.GGUFTokenizerLoader;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Scanner;
import java.util.stream.IntStream;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/** Pure Java baseline with block architecture and Tensor/MemoryView-backed state. */
public final class VanillaLlama {

    private static final String LLAMA3_PATTERN =
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r"
                    + "\\n"
                    + "\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r"
                    + "\\n"
                    + "]*|\\s*[\\r"
                    + "\\n"
                    + "]+|\\s+(?!\\S)|\\s+";
    private static final String GENERAL_ARCHITECTURE = "general.architecture";
    private static final String LLAMA_ARCH = "llama";
    private static final String TOKEN_EMBD_WEIGHT = "token_embd.weight";
    private static final String OUTPUT_WEIGHT = "output.weight";
    private static final String OUTPUT_NORM_WEIGHT = "output_norm.weight";
    private static final String ROPE_FREQS_WEIGHT = "rope_freqs.weight";

    private static final String TOKENIZER_TOKENS = "tokenizer.ggml.tokens";
    private static final String TOKENIZER_TOKEN_TYPE = "tokenizer.ggml.token_type";
    private static final String TOKENIZER_MERGES = "tokenizer.ggml.merges";

    private static final String CFG_EMBEDDING_LENGTH = LLAMA_ARCH + ".embedding_length";
    private static final String CFG_HEAD_COUNT = LLAMA_ARCH + ".attention.head_count";
    private static final String CFG_HEAD_COUNT_KV = LLAMA_ARCH + ".attention.head_count_kv";
    private static final String CFG_FFN_LENGTH = LLAMA_ARCH + ".feed_forward_length";
    private static final String CFG_BLOCK_COUNT = LLAMA_ARCH + ".block_count";
    private static final String CFG_CONTEXT_LENGTH = LLAMA_ARCH + ".context_length";
    private static final String CFG_RMS_EPS = LLAMA_ARCH + ".attention.layer_norm_rms_epsilon";
    private static final String CFG_ROPE_THETA = LLAMA_ARCH + ".rope.freq_base";

    private VanillaLlama() {}

    // ---- CLI ----------------------------------------------------------------------------------

    public static void main(String[] args) throws Exception {
        Options options = Options.parse(args);
        Environment env = Environment.withDefaultDevice(options.device);
        Environment.with(
                env,
                () -> {
                    try (LoadedModel loaded = LoadedModel.load(options.modelPath)) {
                        if (options.interactive) {
                            runInteractive(options, loaded);
                        } else {
                            runInstruct(options, loaded);
                        }
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                    return null;
                });
    }

    private static void runInstruct(Options options, LoadedModel loaded) {
        IntSequence prompt =
                buildInstructPrompt(loaded.chatFormat, options.prompt, options.systemPrompt);
        PrefillResult prefill = prefillWithTiming(loaded.model, prompt);

        Sampler sampler = createSampler(loaded.configuration, options);

        loaded.model.resetTimings();
        long decodeStartNs = System.nanoTime();
        DecodeResult decode =
                decodeTokens(
                        prefill.logits(),
                        options.maxTokens,
                        options.stream,
                        loaded.chatFormat,
                        sampler,
                        loaded.model::computeLogits);
        long decodeElapsedNs = System.nanoTime() - decodeStartNs;
        LlamaModel.TimingSnapshot decodeTimings = loaded.model.timingSnapshot();
        if (options.stream) {
            System.out.println();
        } else {
            System.out.println(loaded.chatFormat.stream(decode.generated()));
        }
        printPerf(
                loaded.model,
                loaded.configuration,
                prefill.tokens(),
                prefill.elapsedNs(),
                prefill.timings(),
                decode.tokens(),
                decodeElapsedNs,
                decodeTimings);
    }

    private static void runInteractive(Options options, LoadedModel loaded) {
        Sampler sampler = createSampler(loaded.configuration, options);

        System.out.println("VanillaLlama chat mode. Type /exit to quit.");
        boolean firstTurn = true;
        try (Scanner scanner = new Scanner(System.in)) {
            while (true) {
                System.out.print("\n> ");
                if (!scanner.hasNextLine()) {
                    break;
                }
                String user = scanner.nextLine();
                if ("/exit".equalsIgnoreCase(user.trim())) {
                    break;
                }
                if (user.isBlank()) {
                    continue;
                }

                IntSequence turn =
                        buildInteractivePrompt(
                                loaded.chatFormat,
                                user,
                                firstTurn ? options.systemPrompt : null,
                                firstTurn);
                firstTurn = false;

                PrefillResult prefill = prefillWithTiming(loaded.model, turn);

                loaded.model.resetTimings();
                long decodeStartNs = System.nanoTime();
                DecodeResult decode =
                        decodeTokens(
                                prefill.logits(),
                                options.maxTokens,
                                true,
                                loaded.chatFormat,
                                sampler,
                                loaded.model::computeLogits);
                long decodeElapsedNs = System.nanoTime() - decodeStartNs;
                LlamaModel.TimingSnapshot decodeTimings = loaded.model.timingSnapshot();
                System.out.println();
                printPerf(
                        loaded.model,
                        loaded.configuration,
                        prefill.tokens(),
                        prefill.elapsedNs(),
                        prefill.timings(),
                        decode.tokens(),
                        decodeElapsedNs,
                        decodeTimings);

                if (loaded.model.isContextFull()) {
                    System.out.println("[context full]");
                    break;
                }
            }
        }
    }

    private static IntSequence buildPrompt(
            Llama3ChatFormat format,
            String userText,
            String systemText,
            boolean includeBeginOfText,
            boolean includeAssistantHeader) {
        IntSequence.Builder out = IntSequence.newBuilder();
        if (includeBeginOfText && format.beginOfText().isPresent()) {
            out.add(format.beginOfText().getAsInt());
        }
        if (systemText != null && !systemText.isBlank()) {
            out.addAll(format.encodeMessage(new Message(Llama3ChatFormat.SYSTEM, systemText)));
        }
        out.addAll(format.encodeMessage(new Message(Llama3ChatFormat.USER, userText)));
        if (includeAssistantHeader) {
            out.addAll(format.encodeHeader(Llama3ChatFormat.ASSISTANT));
        }
        return out.build();
    }

    private static IntSequence buildInstructPrompt(
            Llama3ChatFormat format, String userText, String systemText) {
        return buildPrompt(format, userText, systemText, true, true);
    }

    private static IntSequence buildInteractivePrompt(
            Llama3ChatFormat format, String userText, String systemText, boolean firstTurn) {
        return buildPrompt(format, userText, systemText, firstTurn, true);
    }

    private static Sampler createSampler(Configuration configuration, Options options) {
        return new Sampler(
                configuration.vocabularySize(), options.temperature, options.topP, options.seed);
    }

    private static double tokensPerSecond(int tokens, long elapsedNs) {
        if (tokens <= 0) {
            return 0.0;
        }
        double seconds = Math.max(1L, elapsedNs) / 1_000_000_000.0;
        return tokens / seconds;
    }

    private static String formatTimingReport(
            String label, LlamaModel.TimingSnapshot s, int tokens) {
        double gemvMs = s.gemvNs() / 1_000_000.0;
        double normMs = s.normNs() / 1_000_000.0;
        double ropeMs = s.ropeNs() / 1_000_000.0;
        double attnMs = s.attentionNs() / 1_000_000.0;
        double ffnMs = s.ffnNs() / 1_000_000.0;
        double totalMs = s.totalNs() / 1_000_000.0;
        double msPerToken = tokens > 0 ? totalMs / tokens : 0.0;
        return String.format(
                Locale.ROOT,
                "timings[%s]: gemv=%.2fms norm=%.2fms rope=%.2fms attn=%.2fms ffn=%.2fms"
                        + " total=%.2fms (%.2fms/tok)",
                label,
                gemvMs,
                normMs,
                ropeMs,
                attnMs,
                ffnMs,
                totalMs,
                msPerToken);
    }

    private static void printPerf(
            LlamaModel model,
            Configuration config,
            int prefillTokens,
            long prefillElapsedNs,
            LlamaModel.TimingSnapshot prefillTimings,
            int decodeTokens,
            long decodeElapsedNs,
            LlamaModel.TimingSnapshot decodeTimings) {
        System.out.printf(
                Locale.ROOT, "context: %d/%d%n", model.contextPosition(), config.contextLength());
        System.out.printf(
                Locale.ROOT,
                "perf: prefill %.2f tok/s (%d) decode %.2f tok/s (%d)%n",
                tokensPerSecond(prefillTokens, prefillElapsedNs),
                prefillTokens,
                tokensPerSecond(decodeTokens, decodeElapsedNs),
                decodeTokens);
        System.out.println(formatTimingReport("prefill", prefillTimings, prefillTokens));
        System.out.println(formatTimingReport("decode", decodeTimings, decodeTokens));
    }

    @FunctionalInterface
    private interface DecodeStep {
        MemoryView<?> next(int token);
    }

    private record DecodeResult(IntSequence generated, int tokens) {}

    private record PrefillResult(
            MemoryView<?> logits, int tokens, long elapsedNs, LlamaModel.TimingSnapshot timings) {}

    private static PrefillResult prefillWithTiming(LlamaModel model, IntSequence tokens) {
        int[] tokenArray = tokens.toArray();
        model.resetTimings();
        long startNs = System.nanoTime();
        MemoryView<?> logits = model.ingestTokens(tokenArray);
        long elapsedNs = System.nanoTime() - startNs;
        return new PrefillResult(logits, tokenArray.length, elapsedNs, model.timingSnapshot());
    }

    private static DecodeResult decodeTokens(
            MemoryView<?> logits,
            int maxTokens,
            boolean stream,
            Llama3ChatFormat format,
            Sampler sampler,
            DecodeStep step) {
        IntSequence.Builder generated = IntSequence.newBuilder(maxTokens);
        int produced = 0;
        for (int i = 0; i < maxTokens; i++) {
            int next = sampler.sample(logits);
            if (format.stopTokens().contains(next)) {
                break;
            }
            generated.add(next);
            produced++;
            if (stream) {
                System.out.print(format.stream(IntSequence.of(next)));
                System.out.flush();
            }
            logits = step.next(next);
        }
        return new DecodeResult(generated.build(), produced);
    }

    private record Message(Llama3ChatFormat.Role role, String text)
            implements Llama3ChatFormat.MessageLike {}

    // ---- Model data ---------------------------------------------------------------------------

    record Configuration(
            int dim,
            int ffnDim,
            int nLayers,
            int nHeads,
            int nKvHeads,
            int headDim,
            int contextLength,
            int vocabularySize,
            float rmsEps,
            float ropeTheta) {}

    record LayerWeights(
            Tensor attnNorm,
            Tensor wq,
            Tensor wk,
            Tensor wv,
            Tensor wo,
            Tensor ffnNorm,
            Tensor wGate,
            Tensor wDown,
            Tensor wUp) {}

    record Weights(
            Tensor tokenTable,
            Tensor output,
            Tensor outputNorm,
            LayerWeights[] layers,
            RopeTables rope) {}

    record RopeTables(float[][] cos, float[][] sin) {
        static RopeTables precompute(
                int contextLength, int headDim, float theta, float[] ropeScales) {
            int half = headDim / 2;
            float[][] cos = new float[contextLength][half];
            float[][] sin = new float[contextLength][half];
            for (int p = 0; p < contextLength; p++) {
                for (int i = 0; i < headDim; i += 2) {
                    int f = i / 2;
                    float freq = (float) (1.0 / Math.pow(theta, i / (double) headDim));
                    if (ropeScales != null) {
                        freq /= ropeScales[f];
                    }
                    float v = p * freq;
                    cos[p][f] = (float) Math.cos(v);
                    sin[p][f] = (float) Math.sin(v);
                }
            }
            return new RopeTables(cos, sin);
        }
    }

    static final class State {
        final MemoryView<?>[] keyCache;
        final MemoryView<?>[] valueCache;
        final Scratch scratch;
        int position;

        State(Configuration configuration, MemoryDomain<?> domain) {
            int kvDim = configuration.nKvHeads() * configuration.headDim();
            this.keyCache = new MemoryView<?>[configuration.nLayers()];
            this.valueCache = new MemoryView<?>[configuration.nLayers()];
            for (int l = 0; l < configuration.nLayers(); l++) {
                this.keyCache[l] = alloc(domain, Shape.of(configuration.contextLength(), kvDim));
                this.valueCache[l] = alloc(domain, Shape.of(configuration.contextLength(), kvDim));
            }
            this.scratch = new Scratch(configuration, domain);
            this.position = 0;
        }
    }

    static final class Scratch {
        final MemoryView<?> residual;
        final MemoryView<?> attnInput;
        final MemoryView<?> ffnInput;
        final MemoryView<?> q;
        final MemoryView<?> k;
        final MemoryView<?> v;
        final MemoryView<?> attnMerged;
        final MemoryView<?> attnOutput;
        final MemoryView<?> gate;
        final MemoryView<?> up;
        final MemoryView<?> hidden;
        final MemoryView<?> ffnOutput;
        final MemoryView<?> logits;
        final float[] scores;

        Scratch(Configuration configuration, MemoryDomain<?> domain) {
            int dim = configuration.dim();
            int ffnDim = configuration.ffnDim();
            int kvDim = configuration.nKvHeads() * configuration.headDim();
            this.residual = alloc(domain, Shape.flat(dim));
            this.attnInput = alloc(domain, Shape.flat(dim));
            this.ffnInput = alloc(domain, Shape.flat(dim));
            this.q = alloc(domain, Shape.flat(dim));
            this.k = alloc(domain, Shape.flat(kvDim));
            this.v = alloc(domain, Shape.flat(kvDim));
            this.attnMerged = alloc(domain, Shape.flat(dim));
            this.attnOutput = alloc(domain, Shape.flat(dim));
            this.gate = alloc(domain, Shape.flat(ffnDim));
            this.up = alloc(domain, Shape.flat(ffnDim));
            this.hidden = alloc(domain, Shape.flat(ffnDim));
            this.ffnOutput = alloc(domain, Shape.flat(dim));
            this.logits = alloc(domain, Shape.flat(configuration.vocabularySize()));
            this.scores = new float[configuration.contextLength()];
        }
    }

    // ---- Model execution ----------------------------------------------------------------------

    static final class LlamaModel {
        private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
        private static final ByteOrder NATIVE_ORDER = ByteOrder.nativeOrder();
        private static final int ATTN_TILE_TOKENS = 128;
        private static final boolean USE_FLASHLIKE_DECODE_ATTENTION = true;

        record TimingSnapshot(long gemvNs, long normNs, long ropeNs, long attentionNs, long ffnNs) {
            long totalNs() {
                return gemvNs + normNs + ropeNs + attentionNs + ffnNs;
            }
        }

        private enum TimingKey {
            GEMV,
            NORM,
            ROPE,
            ATTENTION,
            FFN
        }

        private static final class OpTimer {
            private long gemvNs;
            private long normNs;
            private long ropeNs;
            private long attentionNs;
            private long ffnNs;

            Scope scope(TimingKey key) {
                return new Scope(this, key, System.nanoTime());
            }

            void add(TimingKey key, long elapsedNs) {
                switch (key) {
                    case GEMV -> gemvNs += elapsedNs;
                    case NORM -> normNs += elapsedNs;
                    case ROPE -> ropeNs += elapsedNs;
                    case ATTENTION -> attentionNs += elapsedNs;
                    case FFN -> ffnNs += elapsedNs;
                }
            }

            void reset() {
                gemvNs = 0L;
                normNs = 0L;
                ropeNs = 0L;
                attentionNs = 0L;
                ffnNs = 0L;
            }

            TimingSnapshot snapshot() {
                return new TimingSnapshot(gemvNs, normNs, ropeNs, attentionNs, ffnNs);
            }

            static final class Scope implements AutoCloseable {
                private final OpTimer timer;
                private final TimingKey key;
                private final long start;

                Scope(OpTimer timer, TimingKey key, long start) {
                    this.timer = timer;
                    this.key = key;
                    this.start = start;
                }

                @Override
                public void close() {
                    timer.add(key, System.nanoTime() - start);
                }
            }
        }

        private final Configuration configuration;
        private final Weights weights;
        private final MemoryDomain<?> domain;
        private final MemoryAccess<MemorySegment> access;
        private final LayerPlan[] layerPlans;
        private final MemoryView<?> tokenTable;
        private final MemoryView<?> output;
        private final MemoryView<?> outputNorm;
        private final State state;
        private final OpTimer timings = new OpTimer();

        private final NormBlock normBlock = new RmsNormBlock();
        private final AttentionBlock attentionBlock =
                USE_FLASHLIKE_DECODE_ATTENTION
                        ? new CpuFlashLikeDecodeAttentionBlock()
                        : new VanillaAttentionBlock();
        private final FfnBlock ffnBlock = new VanillaSwiGluFfnBlock();
        private final TransformerLayerBlock transformerLayerBlock =
                new VanillaTransformerLayerBlock();

        private interface NormBlock {
            void apply(MemoryView<?> input, MemoryView<?> weight, float eps, MemoryView<?> out);
        }

        private interface AttentionBlock {
            void apply(MemoryView<?> input, LayerPlan plan, int layerIndex, MemoryView<?> out);
        }

        private interface FfnBlock {
            void apply(MemoryView<?> input, LayerPlan plan, Scratch scratch, MemoryView<?> out);
        }

        private interface TransformerLayerBlock {
            void apply(MemoryView<?> x, LayerPlan plan, int layerIndex);
        }

        private record LayerPlan(
                MemoryView<?> attnNorm,
                MemoryView<?> wq,
                MemoryView<?> wk,
                MemoryView<?> wv,
                MemoryView<?> wo,
                MemoryView<?> ffnNorm,
                MemoryView<?> wGate,
                MemoryView<?> wDown,
                MemoryView<?> wUp) {
            static LayerPlan from(LayerWeights w) {
                return new LayerPlan(
                        w.attnNorm().materialize(),
                        w.wq().materialize(),
                        w.wk().materialize(),
                        w.wv().materialize(),
                        w.wo().materialize(),
                        w.ffnNorm().materialize(),
                        w.wGate().materialize(),
                        w.wDown().materialize(),
                        w.wUp().materialize());
            }
        }

        LlamaModel(Configuration configuration, Weights weights) {
            this.configuration = configuration;
            this.weights = weights;
            this.domain =
                    Environment.runtimeFor(Environment.current().defaultDevice()).memoryDomain();
            @SuppressWarnings("unchecked")
            MemoryAccess<MemorySegment> typedAccess =
                    (MemoryAccess<MemorySegment>) domain.directAccess();
            this.access = typedAccess;
            if (this.access == null) {
                throw new IllegalStateException(
                        "No direct memory access for device "
                                + Environment.current().defaultDevice());
            }
            this.tokenTable = weights.tokenTable().materialize();
            this.output = weights.output().materialize();
            this.outputNorm = weights.outputNorm().materialize();
            this.state = new State(configuration, domain);

            this.layerPlans = new LayerPlan[configuration.nLayers()];
            for (int i = 0; i < configuration.nLayers(); i++) {
                this.layerPlans[i] = LayerPlan.from(weights.layers()[i]);
            }
        }

        int contextPosition() {
            return state.position;
        }

        boolean isContextFull() {
            return state.position >= configuration.contextLength() - 1;
        }

        void resetTimings() {
            timings.reset();
        }

        TimingSnapshot timingSnapshot() {
            return timings.snapshot();
        }

        MemoryView<?> ingestTokens(int[] tokens) {
            if (tokens.length == 0) {
                throw new IllegalArgumentException("prefill requires at least one token");
            }
            MemoryView<?> logits = null;
            for (int token : tokens) {
                logits = forwardStep(token);
            }
            return logits;
        }

        MemoryView<?> computeLogits(int token) {
            return forwardStep(token);
        }

        private MemoryView<?> forwardStep(int token) {
            if (state.position >= configuration.contextLength()) {
                throw new IllegalStateException("Context length exceeded");
            }
            if (token < 0 || token >= configuration.vocabularySize()) {
                throw new IllegalArgumentException("Token out of range: " + token);
            }

            Scratch scratch = state.scratch;
            copyRow(tokenTable, token, configuration.dim(), scratch.residual);

            for (int layer = 0; layer < configuration.nLayers(); layer++) {
                transformerLayerBlock.apply(scratch.residual, layerPlans[layer], layer);
            }

            normBlock.apply(
                    scratch.residual, outputNorm, configuration.rmsEps(), scratch.attnInput);
            matVec(
                    output,
                    scratch.attnInput,
                    scratch.logits,
                    configuration.vocabularySize(),
                    configuration.dim());
            state.position++;
            return scratch.logits;
        }

        private final class RmsNormBlock implements NormBlock {
            @Override
            public void apply(
                    MemoryView<?> input, MemoryView<?> weight, float eps, MemoryView<?> out) {
                try (OpTimer.Scope ignored = timings.scope(TimingKey.NORM)) {
                    int n = configuration.dim();
                    float meanSquare = 0f;
                    for (int i = 0; i < n; i++) {
                        float v = read(input, i);
                        meanSquare += v * v;
                    }
                    meanSquare /= n;
                    float scale = (float) (1.0 / Math.sqrt(meanSquare + eps));
                    for (int i = 0; i < n; i++) {
                        write(out, i, read(input, i) * scale * read(weight, i));
                    }
                }
            }
        }

        private final class VanillaAttentionBlock implements AttentionBlock {
            @Override
            public void apply(
                    MemoryView<?> input, LayerPlan plan, int layerIndex, MemoryView<?> out) {
                Scratch scratch = state.scratch;

                matVec(plan.wq(), input, scratch.q, configuration.dim(), configuration.dim());
                matVec(
                        plan.wk(),
                        input,
                        scratch.k,
                        configuration.nKvHeads() * configuration.headDim(),
                        configuration.dim());
                matVec(
                        plan.wv(),
                        input,
                        scratch.v,
                        configuration.nKvHeads() * configuration.headDim(),
                        configuration.dim());

                try (OpTimer.Scope ignored = timings.scope(TimingKey.ROPE)) {
                    applyRope(
                            scratch.q,
                            configuration.nHeads(),
                            configuration.headDim(),
                            state.position,
                            weights.rope());
                    applyRope(
                            scratch.k,
                            configuration.nKvHeads(),
                            configuration.headDim(),
                            state.position,
                            weights.rope());
                }

                int kvDim = configuration.nKvHeads() * configuration.headDim();
                int cacheOffset = state.position * kvDim;
                long kvBytes = (long) kvDim * Float.BYTES;
                MemorySegment.copy(
                        asSegment(scratch.k),
                        scratch.k.byteOffset(),
                        asSegment(state.keyCache[layerIndex]),
                        state.keyCache[layerIndex].byteOffset() + (long) cacheOffset * Float.BYTES,
                        kvBytes);
                MemorySegment.copy(
                        asSegment(scratch.v),
                        scratch.v.byteOffset(),
                        asSegment(state.valueCache[layerIndex]),
                        state.valueCache[layerIndex].byteOffset()
                                + (long) cacheOffset * Float.BYTES,
                        kvBytes);

                try (OpTimer.Scope ignored = timings.scope(TimingKey.ATTENTION)) {
                    int headDim = configuration.headDim();
                    int headUpper = SPECIES.loopBound(headDim);
                    MemorySegment qSeg = asSegment(scratch.q);
                    long qBaseOffset = scratch.q.byteOffset();
                    MemorySegment kCacheSeg = asSegment(state.keyCache[layerIndex]);
                    long kCacheBaseOffset = state.keyCache[layerIndex].byteOffset();
                    MemorySegment vCacheSeg = asSegment(state.valueCache[layerIndex]);
                    long vCacheBaseOffset = state.valueCache[layerIndex].byteOffset();
                    MemorySegment attnSeg = asSegment(scratch.attnMerged);
                    long attnBaseOffset = scratch.attnMerged.byteOffset();

                    int kvHeadGroup = configuration.nHeads() / configuration.nKvHeads();
                    float invSqrtHeadDim = (float) (1.0 / Math.sqrt(headDim));
                    int seqPos = state.position;

                    for (int head = 0; head < configuration.nHeads(); head++) {
                        int qBase = head * headDim;
                        int kvHead = head / kvHeadGroup;
                        int kvBase = kvHead * headDim;
                        long qHeadOffset = qBaseOffset + (long) qBase * Float.BYTES;

                        float max =
                                computeScoresForHead(
                                        scratch.scores,
                                        seqPos,
                                        kvDim,
                                        kvBase,
                                        headDim,
                                        headUpper,
                                        qSeg,
                                        qHeadOffset,
                                        kCacheSeg,
                                        kCacheBaseOffset,
                                        invSqrtHeadDim);

                        float sum = 0f;
                        for (int t = 0; t <= seqPos; t++) {
                            float v = (float) Math.exp(scratch.scores[t] - max);
                            scratch.scores[t] = v;
                            sum += v;
                        }
                        float inv = 1f / sum;

                        writeHeadValues(
                                scratch.scores,
                                seqPos,
                                kvDim,
                                kvBase,
                                headDim,
                                headUpper,
                                inv,
                                vCacheSeg,
                                vCacheBaseOffset,
                                attnSeg,
                                attnBaseOffset,
                                qBase);
                    }
                }

                matVec(
                        plan.wo(),
                        scratch.attnMerged,
                        out,
                        configuration.dim(),
                        configuration.dim());
            }
        }

        private final class CpuFlashLikeDecodeAttentionBlock implements AttentionBlock {
            @Override
            public void apply(
                    MemoryView<?> input, LayerPlan plan, int layerIndex, MemoryView<?> out) {
                Scratch scratch = state.scratch;

                matVec(plan.wq(), input, scratch.q, configuration.dim(), configuration.dim());
                matVec(
                        plan.wk(),
                        input,
                        scratch.k,
                        configuration.nKvHeads() * configuration.headDim(),
                        configuration.dim());
                matVec(
                        plan.wv(),
                        input,
                        scratch.v,
                        configuration.nKvHeads() * configuration.headDim(),
                        configuration.dim());

                try (OpTimer.Scope ignored = timings.scope(TimingKey.ROPE)) {
                    applyRope(
                            scratch.q,
                            configuration.nHeads(),
                            configuration.headDim(),
                            state.position,
                            weights.rope());
                    applyRope(
                            scratch.k,
                            configuration.nKvHeads(),
                            configuration.headDim(),
                            state.position,
                            weights.rope());
                }

                int kvDim = configuration.nKvHeads() * configuration.headDim();
                int cacheOffset = state.position * kvDim;
                long kvBytes = (long) kvDim * Float.BYTES;
                MemorySegment.copy(
                        asSegment(scratch.k),
                        scratch.k.byteOffset(),
                        asSegment(state.keyCache[layerIndex]),
                        state.keyCache[layerIndex].byteOffset() + (long) cacheOffset * Float.BYTES,
                        kvBytes);
                MemorySegment.copy(
                        asSegment(scratch.v),
                        scratch.v.byteOffset(),
                        asSegment(state.valueCache[layerIndex]),
                        state.valueCache[layerIndex].byteOffset()
                                + (long) cacheOffset * Float.BYTES,
                        kvBytes);

                try (OpTimer.Scope ignored = timings.scope(TimingKey.ATTENTION)) {
                    int headDim = configuration.headDim();
                    int headUpper = SPECIES.loopBound(headDim);
                    int seqPos = state.position;
                    int kvHeads = configuration.nKvHeads();
                    int groupSize = configuration.nHeads() / kvHeads;
                    float invSqrtHeadDim = (float) (1.0 / Math.sqrt(headDim));

                    MemorySegment qSeg = asSegment(scratch.q);
                    long qBaseOffset = scratch.q.byteOffset();
                    MemorySegment kCacheSeg = asSegment(state.keyCache[layerIndex]);
                    long kCacheBaseOffset = state.keyCache[layerIndex].byteOffset();
                    MemorySegment vCacheSeg = asSegment(state.valueCache[layerIndex]);
                    long vCacheBaseOffset = state.valueCache[layerIndex].byteOffset();
                    MemorySegment attnSeg = asSegment(scratch.attnMerged);
                    long attnBaseOffset = scratch.attnMerged.byteOffset();

                    IntStream.range(0, kvHeads)
                            .parallel()
                            .forEach(
                                    kvHead -> {
                                        int kvBase = kvHead * headDim;
                                        int headStart = kvHead * groupSize;
                                        float[] tileScores = new float[ATTN_TILE_TOKENS];

                                        for (int g = 0; g < groupSize; g++) {
                                            int head = headStart + g;
                                            int qBase = head * headDim;
                                            long qHeadOffset =
                                                    qBaseOffset + (long) qBase * Float.BYTES;
                                            long outHeadOffset =
                                                    attnBaseOffset + (long) qBase * Float.BYTES;

                                            zeroVector(attnSeg, outHeadOffset, headDim, headUpper);

                                            float m = Float.NEGATIVE_INFINITY;
                                            float l = 0f;

                                            for (int t0 = 0; t0 <= seqPos; t0 += ATTN_TILE_TOKENS) {
                                                int t1 =
                                                        Math.min(seqPos + 1, t0 + ATTN_TILE_TOKENS);
                                                int tileLen = t1 - t0;

                                                float tileMax = Float.NEGATIVE_INFINITY;
                                                for (int t = t0; t < t1; t++) {
                                                    long kOffset =
                                                            kCacheBaseOffset
                                                                    + (long) (t * kvDim + kvBase)
                                                                            * Float.BYTES;
                                                    float score =
                                                            dotHead(
                                                                            qSeg,
                                                                            qHeadOffset,
                                                                            kCacheSeg,
                                                                            kOffset,
                                                                            headDim,
                                                                            headUpper)
                                                                    * invSqrtHeadDim;
                                                    tileScores[t - t0] = score;
                                                    if (score > tileMax) {
                                                        tileMax = score;
                                                    }
                                                }

                                                float mNew = Math.max(m, tileMax);
                                                float alpha =
                                                        (m == Float.NEGATIVE_INFINITY)
                                                                ? 0f
                                                                : (float) Math.exp(m - mNew);
                                                if (alpha != 1f) {
                                                    scaleVector(
                                                            attnSeg,
                                                            outHeadOffset,
                                                            headDim,
                                                            headUpper,
                                                            alpha);
                                                }
                                                l *= alpha;

                                                for (int i = 0; i < tileLen; i++) {
                                                    int t = t0 + i;
                                                    float p =
                                                            (float) Math.exp(tileScores[i] - mNew);
                                                    l += p;
                                                    long vOffset =
                                                            vCacheBaseOffset
                                                                    + (long) (t * kvDim + kvBase)
                                                                            * Float.BYTES;
                                                    axpyVector(
                                                            attnSeg,
                                                            outHeadOffset,
                                                            vCacheSeg,
                                                            vOffset,
                                                            headDim,
                                                            headUpper,
                                                            p);
                                                }
                                                m = mNew;
                                            }

                                            if (l > 0f) {
                                                scaleVector(
                                                        attnSeg,
                                                        outHeadOffset,
                                                        headDim,
                                                        headUpper,
                                                        1f / l);
                                            }
                                        }
                                    });
                }

                matVec(
                        plan.wo(),
                        scratch.attnMerged,
                        out,
                        configuration.dim(),
                        configuration.dim());
            }
        }

        private float dotHead(
                MemorySegment qSeg,
                long qOffset,
                MemorySegment kSeg,
                long kOffset,
                int headDim,
                int headUpper) {
            FloatVector dotAcc = FloatVector.zero(SPECIES);
            int d = 0;
            for (; d < headUpper; d += SPECIES.length()) {
                long qOff = qOffset + (long) d * Float.BYTES;
                long kOff = kOffset + (long) d * Float.BYTES;
                FloatVector qv = FloatVector.fromMemorySegment(SPECIES, qSeg, qOff, NATIVE_ORDER);
                FloatVector kv = FloatVector.fromMemorySegment(SPECIES, kSeg, kOff, NATIVE_ORDER);
                dotAcc = qv.fma(kv, dotAcc);
            }
            float dot = dotAcc.reduceLanes(VectorOperators.ADD);
            for (; d < headDim; d++) {
                long qOff = qOffset + (long) d * Float.BYTES;
                long kOff = kOffset + (long) d * Float.BYTES;
                dot +=
                        qSeg.get(ValueLayout.JAVA_FLOAT_UNALIGNED, qOff)
                                * kSeg.get(ValueLayout.JAVA_FLOAT_UNALIGNED, kOff);
            }
            return dot;
        }

        private void zeroVector(MemorySegment seg, long offset, int dim, int upper) {
            int d = 0;
            FloatVector z = FloatVector.zero(SPECIES);
            for (; d < upper; d += SPECIES.length()) {
                z.intoMemorySegment(seg, offset + (long) d * Float.BYTES, NATIVE_ORDER);
            }
            for (; d < dim; d++) {
                seg.set(ValueLayout.JAVA_FLOAT_UNALIGNED, offset + (long) d * Float.BYTES, 0f);
            }
        }

        private void scaleVector(MemorySegment seg, long offset, int dim, int upper, float scale) {
            FloatVector sv = FloatVector.broadcast(SPECIES, scale);
            int d = 0;
            for (; d < upper; d += SPECIES.length()) {
                long off = offset + (long) d * Float.BYTES;
                FloatVector v = FloatVector.fromMemorySegment(SPECIES, seg, off, NATIVE_ORDER);
                v.mul(sv).intoMemorySegment(seg, off, NATIVE_ORDER);
            }
            for (; d < dim; d++) {
                long off = offset + (long) d * Float.BYTES;
                seg.set(
                        ValueLayout.JAVA_FLOAT_UNALIGNED,
                        off,
                        seg.get(ValueLayout.JAVA_FLOAT_UNALIGNED, off) * scale);
            }
        }

        private void axpyVector(
                MemorySegment dstSeg,
                long dstOffset,
                MemorySegment xSeg,
                long xOffset,
                int dim,
                int upper,
                float alpha) {
            FloatVector av = FloatVector.broadcast(SPECIES, alpha);
            int d = 0;
            for (; d < upper; d += SPECIES.length()) {
                long doff = dstOffset + (long) d * Float.BYTES;
                long xoff = xOffset + (long) d * Float.BYTES;
                FloatVector dv = FloatVector.fromMemorySegment(SPECIES, dstSeg, doff, NATIVE_ORDER);
                FloatVector xv = FloatVector.fromMemorySegment(SPECIES, xSeg, xoff, NATIVE_ORDER);
                xv.fma(av, dv).intoMemorySegment(dstSeg, doff, NATIVE_ORDER);
            }
            for (; d < dim; d++) {
                long doff = dstOffset + (long) d * Float.BYTES;
                long xoff = xOffset + (long) d * Float.BYTES;
                float v =
                        dstSeg.get(ValueLayout.JAVA_FLOAT_UNALIGNED, doff)
                                + alpha * xSeg.get(ValueLayout.JAVA_FLOAT_UNALIGNED, xoff);
                dstSeg.set(ValueLayout.JAVA_FLOAT_UNALIGNED, doff, v);
            }
        }

        private float computeScoresForHead(
                float[] scores,
                int seqPos,
                int kvDim,
                int kvBase,
                int headDim,
                int headUpper,
                MemorySegment qSeg,
                long qHeadOffset,
                MemorySegment kCacheSeg,
                long kCacheBaseOffset,
                float invSqrtHeadDim) {
            float max = Float.NEGATIVE_INFINITY;
            for (int t = 0; t <= seqPos; t++) {
                int keyBase = t * kvDim + kvBase;
                long keyOffset = kCacheBaseOffset + (long) keyBase * Float.BYTES;
                FloatVector dotAcc = FloatVector.zero(SPECIES);
                int d = 0;
                for (; d < headUpper; d += SPECIES.length()) {
                    long qOff = qHeadOffset + (long) d * Float.BYTES;
                    long kOff = keyOffset + (long) d * Float.BYTES;
                    FloatVector qv =
                            FloatVector.fromMemorySegment(SPECIES, qSeg, qOff, NATIVE_ORDER);
                    FloatVector kv =
                            FloatVector.fromMemorySegment(SPECIES, kCacheSeg, kOff, NATIVE_ORDER);
                    dotAcc = qv.fma(kv, dotAcc);
                }
                float dot = dotAcc.reduceLanes(VectorOperators.ADD);
                for (; d < headDim; d++) {
                    long qOff = qHeadOffset + (long) d * Float.BYTES;
                    long kOff = keyOffset + (long) d * Float.BYTES;
                    dot +=
                            qSeg.get(ValueLayout.JAVA_FLOAT_UNALIGNED, qOff)
                                    * kCacheSeg.get(ValueLayout.JAVA_FLOAT_UNALIGNED, kOff);
                }
                float score = dot * invSqrtHeadDim;
                scores[t] = score;
                if (score > max) {
                    max = score;
                }
            }
            return max;
        }

        private void writeHeadValues(
                float[] scores,
                int seqPos,
                int kvDim,
                int kvBase,
                int headDim,
                int headUpper,
                float inv,
                MemorySegment vCacheSeg,
                long vCacheBaseOffset,
                MemorySegment outSeg,
                long outBaseOffset,
                int outBaseIndex) {
            int d = 0;
            for (; d < headUpper; d += SPECIES.length()) {
                FloatVector valueAcc = FloatVector.zero(SPECIES);
                for (int t = 0; t <= seqPos; t++) {
                    int valueBase = t * kvDim + kvBase + d;
                    long valueOffset = vCacheBaseOffset + (long) valueBase * Float.BYTES;
                    FloatVector vv =
                            FloatVector.fromMemorySegment(
                                    SPECIES, vCacheSeg, valueOffset, NATIVE_ORDER);
                    FloatVector wv = FloatVector.broadcast(SPECIES, scores[t] * inv);
                    valueAcc = vv.fma(wv, valueAcc);
                }
                long outOffset = outBaseOffset + (long) (outBaseIndex + d) * Float.BYTES;
                valueAcc.intoMemorySegment(outSeg, outOffset, NATIVE_ORDER);
            }
            for (; d < headDim; d++) {
                float acc = 0f;
                for (int t = 0; t <= seqPos; t++) {
                    int valueBase = t * kvDim + kvBase + d;
                    long valueOffset = vCacheBaseOffset + (long) valueBase * Float.BYTES;
                    acc +=
                            scores[t]
                                    * inv
                                    * vCacheSeg.get(ValueLayout.JAVA_FLOAT_UNALIGNED, valueOffset);
                }
                long outOffset = outBaseOffset + (long) (outBaseIndex + d) * Float.BYTES;
                outSeg.set(ValueLayout.JAVA_FLOAT_UNALIGNED, outOffset, acc);
            }
        }

        private final class VanillaSwiGluFfnBlock implements FfnBlock {
            @Override
            public void apply(
                    MemoryView<?> input, LayerPlan plan, Scratch scratch, MemoryView<?> out) {

                matVec(
                        plan.wGate(),
                        input,
                        scratch.gate,
                        configuration.ffnDim(),
                        configuration.dim());
                matVec(plan.wUp(), input, scratch.up, configuration.ffnDim(), configuration.dim());

                try (OpTimer.Scope ignored = timings.scope(TimingKey.FFN)) {
                    for (int i = 0; i < configuration.ffnDim(); i++) {
                        float g = read(scratch.gate, i);
                        float silu = g / (1f + (float) Math.exp(-g));
                        write(scratch.hidden, i, silu * read(scratch.up, i));
                    }
                }

                matVec(
                        plan.wDown(),
                        scratch.hidden,
                        out,
                        configuration.dim(),
                        configuration.ffnDim());
            }
        }

        private final class VanillaTransformerLayerBlock implements TransformerLayerBlock {
            @Override
            public void apply(MemoryView<?> x, LayerPlan plan, int layerIndex) {
                Scratch scratch = state.scratch;

                normBlock.apply(x, plan.attnNorm(), configuration.rmsEps(), scratch.attnInput);
                attentionBlock.apply(scratch.attnInput, plan, layerIndex, scratch.attnOutput);
                addInPlace(x, scratch.attnOutput, configuration.dim());

                normBlock.apply(x, plan.ffnNorm(), configuration.rmsEps(), scratch.ffnInput);
                ffnBlock.apply(scratch.ffnInput, plan, scratch, scratch.ffnOutput);
                addInPlace(x, scratch.ffnOutput, configuration.dim());
            }
        }

        private void matVec(
                MemoryView<?> matrix,
                MemoryView<?> vector,
                MemoryView<?> out,
                int outDim,
                int inDim) {
            try (OpTimer.Scope ignored = timings.scope(TimingKey.GEMV)) {
                MemorySegment mSeg = asSegment(matrix);
                MemorySegment vSeg = asSegment(vector);
                MemorySegment oSeg = asSegment(out);
                long mBase = matrix.byteOffset();
                long vBase = vector.byteOffset();
                long oBase = out.byteOffset();
                int upper = SPECIES.loopBound(inDim);

                IntStream.range(0, outDim)
                        .parallel()
                        .forEach(
                                r ->
                                        matVecRow(
                                                mSeg, vSeg, oSeg, mBase, vBase, oBase, inDim, upper,
                                                r));
            }
        }

        private void matVecRow(
                MemorySegment mSeg,
                MemorySegment vSeg,
                MemorySegment oSeg,
                long mBase,
                long vBase,
                long oBase,
                int inDim,
                int upper,
                int row) {
            long rowBase = mBase + (long) row * inDim * Float.BYTES;
            FloatVector acc = FloatVector.zero(SPECIES);
            int c = 0;
            for (; c < upper; c += SPECIES.length()) {
                long mOff = rowBase + (long) c * Float.BYTES;
                long vOff = vBase + (long) c * Float.BYTES;
                FloatVector mv = FloatVector.fromMemorySegment(SPECIES, mSeg, mOff, NATIVE_ORDER);
                FloatVector vv = FloatVector.fromMemorySegment(SPECIES, vSeg, vOff, NATIVE_ORDER);
                acc = mv.fma(vv, acc);
            }
            float sum = acc.reduceLanes(VectorOperators.ADD);
            for (; c < inDim; c++) {
                long mOff = rowBase + (long) c * Float.BYTES;
                long vOff = vBase + (long) c * Float.BYTES;
                sum +=
                        mSeg.get(ValueLayout.JAVA_FLOAT, mOff)
                                * vSeg.get(ValueLayout.JAVA_FLOAT, vOff);
            }
            long oOff = oBase + (long) row * Float.BYTES;
            oSeg.set(ValueLayout.JAVA_FLOAT_UNALIGNED, oOff, sum);
        }

        @SuppressWarnings("unchecked")
        private static MemorySegment asSegment(MemoryView<?> view) {
            Object base = view.memory().base();
            if (base instanceof MemorySegment segment) {
                return segment;
            }
            throw new IllegalStateException(
                    "Expected Memory<MemorySegment> for panama backend, got "
                            + view.memory().getClass().getName());
        }

        private void copyRow(MemoryView<?> matrix, int row, int rowWidth, MemoryView<?> out) {
            int base = row * rowWidth;
            for (int i = 0; i < rowWidth; i++) {
                write(out, i, read(matrix, base + i));
            }
        }

        private void addInPlace(MemoryView<?> dst, MemoryView<?> src, int length) {
            for (int i = 0; i < length; i++) {
                write(dst, i, read(dst, i) + read(src, i));
            }
        }

        private void applyRope(
                MemoryView<?> tensor,
                int nHeads,
                int headDim,
                int position,
                RopeTables ropeTables) {
            float[] cos = ropeTables.cos()[position];
            float[] sin = ropeTables.sin()[position];
            for (int h = 0; h < nHeads; h++) {
                int base = h * headDim;
                for (int i = 0; i < headDim; i += 2) {
                    int idx0 = base + i;
                    int idx1 = base + i + 1;
                    float x0 = read(tensor, idx0);
                    float x1 = read(tensor, idx1);
                    float c = cos[i / 2];
                    float s = sin[i / 2];
                    write(tensor, idx0, x0 * c - x1 * s);
                    write(tensor, idx1, x0 * s + x1 * c);
                }
            }
        }

        private float read(MemoryView<?> view, int index) {
            return access.readFloat(asMemory(view), view.byteOffset() + (long) index * Float.BYTES);
        }

        private void write(MemoryView<?> view, int index, float value) {
            access.writeFloat(
                    asMemory(view), view.byteOffset() + (long) index * Float.BYTES, value);
        }

        @SuppressWarnings("unchecked")
        private static Memory<MemorySegment> asMemory(MemoryView<?> view) {
            return (Memory<MemorySegment>) view.memory();
        }
    }

    private static final class LoadedModel implements AutoCloseable {
        final Configuration configuration;
        final Tokenizer tokenizer;
        final Llama3ChatFormat chatFormat;
        final LlamaModel model;
        final Arena arena;
        final FileChannel channel;

        private LoadedModel(
                Configuration configuration,
                Tokenizer tokenizer,
                Llama3ChatFormat chatFormat,
                LlamaModel model,
                Arena arena,
                FileChannel channel) {
            this.configuration = configuration;
            this.tokenizer = tokenizer;
            this.chatFormat = chatFormat;
            this.model = model;
            this.arena = arena;
            this.channel = channel;
        }

        static LoadedModel load(Path modelPath) throws IOException {
            GGUF gguf = GGUF.read(modelPath);
            String arch = gguf.getValue(String.class, GENERAL_ARCHITECTURE);
            if (!LLAMA_ARCH.equals(arch)) {
                throw new IllegalArgumentException("Expected llama architecture, got: " + arch);
            }

            Tokenizer tokenizer = loadTokenizer(modelPath);
            Configuration config = loadConfiguration(gguf, tokenizer.vocabulary().size());
            Llama3ChatFormat format = new Llama3ChatFormat(tokenizer);

            FileChannel channel = FileChannel.open(modelPath, StandardOpenOption.READ);
            Arena arena = Arena.ofShared();
            long tensorBase = gguf.getTensorDataOffset();
            Weights weights = loadWeights(gguf, config, tensorBase, channel, arena);

            return new LoadedModel(
                    config, tokenizer, format, new LlamaModel(config, weights), arena, channel);
        }

        @Override
        public void close() throws IOException {
            arena.close();
            channel.close();
        }
    }

    // ---- GGUF loading -------------------------------------------------------------------------

    private static Tokenizer loadTokenizer(Path modelPath) {
        return GGUFTokenizerLoader.createBuilderWithBuiltins().build().fromLocal(modelPath);
    }

    private static Configuration loadConfiguration(GGUF gguf, int vocab) {
        int dim = gguf.getValue(int.class, CFG_EMBEDDING_LENGTH);
        int heads = gguf.getValue(int.class, CFG_HEAD_COUNT);
        int kvHeads = gguf.getValue(int.class, CFG_HEAD_COUNT_KV);
        int headDim = dim / heads;
        return new Configuration(
                dim,
                gguf.getValue(int.class, CFG_FFN_LENGTH),
                gguf.getValue(int.class, CFG_BLOCK_COUNT),
                heads,
                kvHeads,
                headDim,
                gguf.getValue(int.class, CFG_CONTEXT_LENGTH),
                vocab,
                gguf.getValue(float.class, CFG_RMS_EPS),
                gguf.getValue(float.class, CFG_ROPE_THETA));
    }

    private static Weights loadWeights(
            GGUF gguf, Configuration cfg, long tensorBase, FileChannel channel, Arena arena) {
        Tensor tokenTable = loadTensor(gguf, TOKEN_EMBD_WEIGHT, tensorBase, channel, arena);
        Tensor output =
                gguf.containsTensor(OUTPUT_WEIGHT)
                        ? loadTensor(gguf, OUTPUT_WEIGHT, tensorBase, channel, arena)
                        : tokenTable;

        LayerWeights[] layers = new LayerWeights[cfg.nLayers()];
        for (int i = 0; i < cfg.nLayers(); i++) {
            layers[i] = loadLayerWeights(gguf, i, tensorBase, channel, arena);
        }

        Tensor outNorm = loadTensor(gguf, OUTPUT_NORM_WEIGHT, tensorBase, channel, arena);
        float[] ropeScales =
                gguf.containsTensor(ROPE_FREQS_WEIGHT)
                        ? toFloatArray(
                                loadTensor(gguf, ROPE_FREQS_WEIGHT, tensorBase, channel, arena))
                        : null;
        RopeTables rope =
                RopeTables.precompute(
                        cfg.contextLength(), cfg.headDim(), cfg.ropeTheta(), ropeScales);

        return new Weights(tokenTable, output, outNorm, layers, rope);
    }

    private static Tensor loadTensor(
            GGUF gguf, String name, long tensorBase, FileChannel channel, Arena arena) {
        TensorEntry entry = gguf.getTensor(name);
        if (entry == null) {
            throw new IllegalArgumentException("Missing tensor: " + name);
        }
        if (entry.ggmlType() != GGMLType.F32) {
            throw new UnsupportedOperationException(
                    "Only F32 is supported in VanillaLlama; " + name + " is " + entry.ggmlType());
        }

        long bytes = entry.ggmlType().byteSizeFor(entry.totalNumberOfElements());
        try {
            MemorySegment seg =
                    channel.map(
                            FileChannel.MapMode.READ_ONLY,
                            tensorBase + entry.offset(),
                            bytes,
                            arena);
            Memory<MemorySegment> mem = MemoryFactory.ofMemorySegment(seg);
            Shape shape =
                    entry.shape().length == 2
                            ? Shape.of(entry.shape()[1], entry.shape()[0])
                            : Shape.flat(entry.shape());
            MemoryView<MemorySegment> view =
                    MemoryView.of(mem, 0, DataType.FP32, Layout.rowMajor(shape));
            return Tensor.of(view);
        } catch (IOException ex) {
            throw new RuntimeException("Failed to map tensor: " + name, ex);
        }
    }

    private static LayerWeights loadLayerWeights(
            GGUF gguf, int layer, long tensorBase, FileChannel channel, Arena arena) {
        return new LayerWeights(
                loadTensor(
                        gguf,
                        blockTensorName(layer, "attn_norm.weight"),
                        tensorBase,
                        channel,
                        arena),
                loadTensor(
                        gguf, blockTensorName(layer, "attn_q.weight"), tensorBase, channel, arena),
                loadTensor(
                        gguf, blockTensorName(layer, "attn_k.weight"), tensorBase, channel, arena),
                loadTensor(
                        gguf, blockTensorName(layer, "attn_v.weight"), tensorBase, channel, arena),
                loadTensor(
                        gguf,
                        blockTensorName(layer, "attn_output.weight"),
                        tensorBase,
                        channel,
                        arena),
                loadTensor(
                        gguf,
                        blockTensorName(layer, "ffn_norm.weight"),
                        tensorBase,
                        channel,
                        arena),
                loadTensor(
                        gguf,
                        blockTensorName(layer, "ffn_gate.weight"),
                        tensorBase,
                        channel,
                        arena),
                loadTensor(
                        gguf,
                        blockTensorName(layer, "ffn_down.weight"),
                        tensorBase,
                        channel,
                        arena),
                loadTensor(
                        gguf, blockTensorName(layer, "ffn_up.weight"), tensorBase, channel, arena));
    }

    private static float[] toFloatArray(Tensor tensor) {
        MemoryView<?> view = tensor.materialize();
        int size = Math.toIntExact(view.shape().size());
        float[] out = new float[size];

        @SuppressWarnings({"rawtypes", "unchecked"})
        MemoryAccess access =
                Environment.runtimeFor(view.memory().device()).memoryDomain().directAccess();
        for (int i = 0; i < size; i++) {
            out[i] = access.readFloat(view.memory(), view.byteOffset() + (long) i * Float.BYTES);
        }
        return out;
    }

    private static String blockTensorName(int index, String name) {
        return "blk." + index + "." + name;
    }

    // ---- Memory helper ------------------------------------------------------------------------

    private static MemoryView<?> alloc(MemoryDomain<?> domain, Shape shape) {
        return MemoryView.of(
                domain.memoryAllocator().allocateMemory(DataType.FP32, shape),
                DataType.FP32,
                Layout.rowMajor(shape));
    }
}
