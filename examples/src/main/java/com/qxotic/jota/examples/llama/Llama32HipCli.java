package com.qxotic.jota.examples.llama;

import com.qxotic.format.gguf.GGMLType;
import com.qxotic.format.gguf.GGUF;
import com.qxotic.format.gguf.TensorEntry;
import com.qxotic.jota.*;
import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryAccess;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.memory.impl.MemoryFactory;
import com.qxotic.jota.runtime.DeviceRuntime;
import com.qxotic.jota.runtime.ExecutionStream;
import com.qxotic.jota.runtime.KernelArgs;
import com.qxotic.jota.runtime.KernelExecutable;
import com.qxotic.jota.runtime.KernelProgram;
import com.qxotic.jota.runtime.LaunchConfig;
import com.qxotic.jota.tensor.Tensor;
import com.qxotic.tokenizers.IntSequence;
import com.qxotic.tokenizers.Tokenizer;
import com.qxotic.tokenizers.advanced.Normalizer;
import com.qxotic.tokenizers.advanced.Splitter;
import com.qxotic.tokenizers.impl.GPT2Tokenizer;
import com.qxotic.tokenizers.impl.IntPair;
import com.qxotic.tokenizers.impl.RegexSplitter;
import com.qxotic.tokenizers.impl.VocabularyImpl;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Scanner;

public final class Llama32HipCli {
    private static final int TIMING_WARMUP_TOKENS = 2;
    private static final int PAGE_BYTES = 4096;
    private static volatile float touchSink;

    private static final String LLAMA3_PATTERN =
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r"
                    + "\\n"
                    + "\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r"
                    + "\\n"
                    + "]*|\\s*[\\r"
                    + "\\n"
                    + "]+|\\s+(?!\\S)|\\s+";

    public static void main(String[] args) throws Exception {
        Options options = Options.parse(args);
        if (!Environment.hasRuntimeFor(DeviceType.HIP.deviceIndex(0))) {
            throw new IllegalStateException(
                    "HIP runtime is not available. Build/load jota-runtime-hip and ROCm/HIP runtime"
                            + " first.");
        }
        Environment env = Environment.withDefaultDevice(DeviceType.HIP.deviceIndex(0));
        Environment.with(
                env,
                () -> {
                    try {
                        run(options);
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                    return null;
                });
    }

    private static void run(Options options) throws IOException {
        if (options.trace) {
            System.setProperty("com.qxotic.trace", "true");
        }
        try (LoadedModel loaded = LoadedModel.load(options.modelPath)) {
            if (options.interactive) {
                runInteractive(options, loaded);
            } else {
                runInstruct(options, loaded);
            }
        }
    }

    private static void runInstruct(Options options, LoadedModel loaded) {
        if (options.prompt == null || options.prompt.isBlank()) {
            throw new IllegalArgumentException("--prompt is required in non-interactive mode");
        }
        Llama3ChatFormat format = loaded.chatFormat;
        Llama32Model model = loaded.model;
        int runs = Math.max(1, options.benchmarkRuns);
        for (int run = 1; run <= runs; run++) {
            if (run > 1 && options.benchmarkPauseMs > 0) {
                System.out.printf(
                        "benchmark pause before run %d/%d: %d ms (pid=%d)%n",
                        run, runs, options.benchmarkPauseMs, ProcessHandle.current().pid());
                try {
                    Thread.sleep(options.benchmarkPauseMs);
                } catch (InterruptedException ex) {
                    Thread.currentThread().interrupt();
                    throw new RuntimeException("Benchmark pause interrupted", ex);
                }
            }
            if (runs > 1) {
                System.out.printf("benchmark run %d/%d%n", run, runs);
            }
            model.resetDecodeState();
            Llama32Model.DecodeScratch scratch = model.createDecodeScratch();
            model.resetTimingStats();
            long wallStartNs = System.nanoTime();
            Sampler sampler =
                    new Sampler(
                            loaded.config.vocabularySize,
                            options.temperature,
                            options.topP,
                            options.seed);

            List<Integer> promptTokens = new ArrayList<>();
            format.beginOfText().ifPresent(promptTokens::add);
            if (options.systemPrompt != null && !options.systemPrompt.isBlank()) {
                append(
                        promptTokens,
                        format.encodeMessage(
                                new Message(Llama3ChatFormat.SYSTEM, options.systemPrompt)));
            }
            append(
                    promptTokens,
                    format.encodeMessage(new Message(Llama3ChatFormat.USER, options.prompt)));
            append(promptTokens, format.encodeHeader(Llama3ChatFormat.ASSISTANT));

            MemoryView<?> logits = null;
            for (int token : promptTokens) {
                if (options.trace) {
                    System.out.printf(
                            "TRACE tensor token phase=ingest pos=%d id=%d%n",
                            model.position, token);
                }
                logits = model.forward(token, scratch);
            }

            IntSequence.Builder generated = IntSequence.newBuilder();
            model.resetTimingStats();
            long generationStartNs = -1L;
            long timedWallStartNs = -1L;
            int generatedCount = 0;
            int timedGeneratedCount = 0;
            long samplingTimeNs = 0L;
            long samplingCalls = 0L;
            long samplingMaxNs = 0L;
            for (int i = 0; i < options.maxTokens; i++) {
                long samplingStartNs = System.nanoTime();
                int next = sampler.sample(logits);
                long samplingElapsedNs = System.nanoTime() - samplingStartNs;
                if (options.trace) {
                    System.out.printf(
                            "TRACE tensor token phase=produce pos=%d id=%d%n",
                            model.position, next);
                }
                generatedCount++;
                if (generatedCount == TIMING_WARMUP_TOKENS + 1) {
                    generationStartNs = System.nanoTime();
                    timedWallStartNs = generationStartNs;
                    model.resetTimingStats();
                    samplingTimeNs = 0L;
                    samplingCalls = 0L;
                    samplingMaxNs = 0L;
                }
                if (generatedCount > TIMING_WARMUP_TOKENS) {
                    timedGeneratedCount++;
                    samplingTimeNs += samplingElapsedNs;
                    samplingCalls++;
                    samplingMaxNs = Math.max(samplingMaxNs, samplingElapsedNs);
                }
                generated.add(next);
                if (options.stream) {
                    System.out.print(format.stream(IntSequence.of(next)));
                }
                logits = model.forward(next, scratch);
                if (format.stopTokens().contains(next)) {
                    break;
                }
            }
            long generationEndNs = System.nanoTime();
            printTokensPerSecond(timedGeneratedCount, generationStartNs, generationEndNs);
            long wallEndNs = System.nanoTime();
            long timedWallNs =
                    (timedWallStartNs > 0 && wallEndNs > timedWallStartNs)
                            ? (wallEndNs - timedWallStartNs)
                            : 0L;
            model.printTimingSummary("instruct", timedWallNs);
            printSamplingTimings("instruct", samplingTimeNs, samplingCalls, samplingMaxNs);

            if (options.stream) {
                System.out.println();
            } else {
                IntSequence out = generated.build();
                if (!out.isEmpty() && format.stopTokens().contains(out.getLast())) {
                    out = out.subSequence(0, out.length() - 1);
                }
                if (run == runs) {
                    System.out.println(format.echo(out));
                }
            }
        }
    }

    private static void runInteractive(Options options, LoadedModel loaded) {
        Llama3ChatFormat format = loaded.chatFormat;
        Llama32Model model = loaded.model;
        Sampler sampler =
                new Sampler(
                        loaded.config.vocabularySize,
                        options.temperature,
                        options.topP,
                        options.seed);

        List<Integer> conversation = new ArrayList<>();
        format.beginOfText().ifPresent(conversation::add);
        if (options.systemPrompt != null && !options.systemPrompt.isBlank()) {
            append(
                    conversation,
                    format.encodeMessage(
                            new Message(Llama3ChatFormat.SYSTEM, options.systemPrompt)));
        }

        int consumed = 0;
        Scanner scanner = new Scanner(System.in);
        while (true) {
            System.out.print("> ");
            String user = scanner.nextLine();
            if ("quit".equalsIgnoreCase(user) || "exit".equalsIgnoreCase(user)) {
                break;
            }

            append(conversation, format.encodeMessage(new Message(Llama3ChatFormat.USER, user)));
            append(conversation, format.encodeHeader(Llama3ChatFormat.ASSISTANT));
            Llama32Model.DecodeScratch scratch = model.createDecodeScratch();
            model.resetTimingStats();
            long wallStartNs = System.nanoTime();

            MemoryView<?> logits = null;
            for (int i = consumed; i < conversation.size(); i++) {
                if (options.trace) {
                    System.out.printf(
                            "TRACE tensor token phase=ingest pos=%d id=%d%n",
                            model.position, conversation.get(i));
                }
                logits = model.forward(conversation.get(i), scratch);
            }
            consumed = conversation.size();

            model.resetTimingStats();
            int generatedCount = 0;
            int timedGeneratedCount = 0;
            long samplingTimeNs = 0L;
            long samplingCalls = 0L;
            long samplingMaxNs = 0L;
            long generationStartNs = -1L;
            long timedWallStartNs = -1L;
            for (int i = 0; i < options.maxTokens; i++) {
                long samplingStartNs = System.nanoTime();
                int next = sampler.sample(logits);
                long samplingElapsedNs = System.nanoTime() - samplingStartNs;
                if (options.trace) {
                    System.out.printf(
                            "TRACE tensor token phase=produce pos=%d id=%d%n",
                            model.position, next);
                }
                conversation.add(next);
                consumed++;
                generatedCount++;
                if (generatedCount == TIMING_WARMUP_TOKENS + 1) {
                    generationStartNs = System.nanoTime();
                    timedWallStartNs = generationStartNs;
                    model.resetTimingStats();
                    samplingTimeNs = 0L;
                    samplingCalls = 0L;
                    samplingMaxNs = 0L;
                }
                if (generatedCount > TIMING_WARMUP_TOKENS) {
                    timedGeneratedCount++;
                    samplingTimeNs += samplingElapsedNs;
                    samplingCalls++;
                    samplingMaxNs = Math.max(samplingMaxNs, samplingElapsedNs);
                }
                if (options.stream) {
                    System.out.print(format.stream(IntSequence.of(next)));
                }
                logits = model.forward(next, scratch);
                if (format.stopTokens().contains(next)) {
                    break;
                }
            }
            long generationEndNs = System.nanoTime();
            if (options.stream) {
                System.out.println();
            }
            printTokensPerSecond(timedGeneratedCount, generationStartNs, generationEndNs);
            long wallEndNs = System.nanoTime();
            long timedWallNs =
                    (timedWallStartNs > 0 && wallEndNs > timedWallStartNs)
                            ? (wallEndNs - timedWallStartNs)
                            : 0L;
            model.printTimingSummary("interactive", timedWallNs);
            printSamplingTimings("interactive", samplingTimeNs, samplingCalls, samplingMaxNs);
        }
    }

    private static void printTokensPerSecond(int tokenCount, long startNs, long endNs) {
        if (tokenCount <= 0 || startNs <= 0 || endNs <= startNs) {
            System.out.println("tokens/s: n/a (0 tokens)");
            return;
        }
        double seconds = (endNs - startNs) / 1_000_000_000.0;
        double tps = tokenCount / seconds;
        System.out.printf("tokens/s: %.2f (%d tokens in %.3fs)%n", tps, tokenCount, seconds);
    }

    private static void printSamplingTimings(
            String mode, long samplingTimeNs, long samplingCalls, long samplingMaxNs) {
        if (samplingCalls <= 0) {
            System.out.printf("sampling[%s]: n/a (0 samples)%n", mode);
            return;
        }
        double totalMs = samplingTimeNs / 1_000_000.0;
        double avgUs = (samplingTimeNs / 1_000.0) / samplingCalls;
        double maxUs = samplingMaxNs / 1_000.0;
        System.out.printf(
                "sampling[%s]: total=%.3fms calls=%d avg=%.3fus max=%.3fus%n",
                mode, totalMs, samplingCalls, avgUs, maxUs);
    }

    private static void append(List<Integer> out, IntSequence seq) {
        for (int i = 0; i < seq.length(); i++) {
            out.add(seq.intAt(i));
        }
    }

    private static final class LoadedModel implements AutoCloseable {
        final LlamaConfig config;
        final Tokenizer tokenizer;
        final Llama3ChatFormat chatFormat;
        final Llama32Model model;
        final Arena arena;
        final FileChannel channel;

        private LoadedModel(
                LlamaConfig config,
                Tokenizer tokenizer,
                Llama3ChatFormat chatFormat,
                Llama32Model model,
                Arena arena,
                FileChannel channel) {
            this.config = config;
            this.tokenizer = tokenizer;
            this.chatFormat = chatFormat;
            this.model = model;
            this.arena = arena;
            this.channel = channel;
        }

        static LoadedModel load(Path modelPath) throws IOException {
            GGUF gguf = GGUF.read(modelPath);
            String arch = gguf.getValue(String.class, "general.architecture");
            if (!"llama".equals(arch)) {
                throw new IllegalArgumentException("Expected llama architecture, got: " + arch);
            }
            Tokenizer tokenizer = loadTokenizer(gguf);
            LlamaConfig config = loadConfig(gguf, tokenizer.vocabulary().size());
            Llama3ChatFormat format = new Llama3ChatFormat(tokenizer);

            FileChannel channel = FileChannel.open(modelPath, StandardOpenOption.READ);
            Arena arena = Arena.ofShared();
            long tensorBase = gguf.getTensorDataOffset();
            LlamaWeights weights = loadWeights(gguf, config, tensorBase, channel, arena);
            return new LoadedModel(
                    config, tokenizer, format, new Llama32Model(config, weights), arena, channel);
        }

        @Override
        public void close() throws IOException {
            arena.close();
            channel.close();
        }
    }

    private static Tokenizer loadTokenizer(GGUF gguf) {
        String[] tokens = gguf.getValue(String[].class, "tokenizer.ggml.tokens");
        int[] tokenTypes =
                gguf.containsKey("tokenizer.ggml.token_type")
                        ? gguf.getValue(int[].class, "tokenizer.ggml.token_type")
                        : null;
        String[] merges = gguf.getValue(String[].class, "tokenizer.ggml.merges");
        VocabularyImpl vocabulary = new VocabularyImpl(tokens, null, tokenTypes);
        List<IntPair> pairs =
                Arrays.stream(merges)
                        .map(s -> s.split(" "))
                        .map(p -> new IntPair(vocabulary.id(p[0]), vocabulary.id(p[1])))
                        .toList();
        Splitter splitter = RegexSplitter.create(LLAMA3_PATTERN);
        return new GPT2Tokenizer(vocabulary, Normalizer.IDENTITY, splitter, pairs);
    }

    private static LlamaConfig loadConfig(GGUF gguf, int vocab) {
        String a = "llama";
        int dim = gguf.getValue(int.class, a + ".embedding_length");
        int heads = gguf.getValue(int.class, a + ".attention.head_count");
        int kvHeads = gguf.getValue(int.class, a + ".attention.head_count_kv");
        int headDim = dim / heads;
        int contextLength = gguf.getValue(int.class, a + ".context_length");
        // Cap context length to reasonable value to avoid huge memory allocations
        int maxContext = 2048;
        if (contextLength > maxContext) {
            System.err.println(
                    "[Config] Capping context length from " + contextLength + " to " + maxContext);
            contextLength = maxContext;
        }
        return new LlamaConfig(
                dim,
                gguf.getValue(int.class, a + ".feed_forward_length"),
                gguf.getValue(int.class, a + ".block_count"),
                heads,
                kvHeads,
                headDim,
                contextLength,
                vocab,
                gguf.getValue(float.class, a + ".attention.layer_norm_rms_epsilon"),
                gguf.getValue(float.class, a + ".rope.freq_base"));
    }

    private static LlamaWeights loadWeights(
            GGUF gguf, LlamaConfig cfg, long tensorBase, FileChannel channel, Arena arena) {
        Tensor tokenTable =
                mapTensorAsArrayTensor(gguf, "token_embd.weight", tensorBase, channel, arena);

        Tensor output =
                gguf.containsTensor("output.weight")
                        ? mapTensorAsArrayTensor(gguf, "output.weight", tensorBase, channel, arena)
                        : tokenTable;

        Tensor[] attnNorm = new Tensor[cfg.nLayers];
        Tensor[] wq = new Tensor[cfg.nLayers];
        Tensor[] wk = new Tensor[cfg.nLayers];
        Tensor[] wv = new Tensor[cfg.nLayers];
        Tensor[] wo = new Tensor[cfg.nLayers];
        Tensor[] ffnNorm = new Tensor[cfg.nLayers];
        Tensor[] wGate = new Tensor[cfg.nLayers];
        Tensor[] wDown = new Tensor[cfg.nLayers];
        Tensor[] wUp = new Tensor[cfg.nLayers];
        for (int i = 0; i < cfg.nLayers; i++) {
            attnNorm[i] =
                    mapTensorAsArrayTensor(
                            gguf, "blk." + i + ".attn_norm.weight", tensorBase, channel, arena);
            wq[i] =
                    mapTensorAsArrayTensor(
                            gguf, "blk." + i + ".attn_q.weight", tensorBase, channel, arena);
            wk[i] =
                    mapTensorAsArrayTensor(
                            gguf, "blk." + i + ".attn_k.weight", tensorBase, channel, arena);
            wv[i] =
                    mapTensorAsArrayTensor(
                            gguf, "blk." + i + ".attn_v.weight", tensorBase, channel, arena);
            wo[i] =
                    mapTensorAsArrayTensor(
                            gguf, "blk." + i + ".attn_output.weight", tensorBase, channel, arena);
            ffnNorm[i] =
                    mapTensorAsArrayTensor(
                            gguf, "blk." + i + ".ffn_norm.weight", tensorBase, channel, arena);
            wGate[i] =
                    mapTensorAsArrayTensor(
                            gguf, "blk." + i + ".ffn_gate.weight", tensorBase, channel, arena);
            wDown[i] =
                    mapTensorAsArrayTensor(
                            gguf, "blk." + i + ".ffn_down.weight", tensorBase, channel, arena);
            wUp[i] =
                    mapTensorAsArrayTensor(
                            gguf, "blk." + i + ".ffn_up.weight", tensorBase, channel, arena);
        }

        Tensor outNorm =
                mapTensorAsArrayTensor(gguf, "output_norm.weight", tensorBase, channel, arena);
        float[] ropeScales =
                gguf.containsTensor("rope_freqs.weight")
                        ? toFloatArray(
                                mapTensorAsArrayTensor(
                                        gguf, "rope_freqs.weight", tensorBase, channel, arena))
                        : null;
        RopeTables rope =
                RopeTables.precompute(
                        cfg.contextLength, cfg.headDim, cfg.ropeTheta, ropeScales, cfg.nHeads);
        return new LlamaWeights(
                tokenTable,
                output,
                outNorm,
                attnNorm,
                wq,
                wk,
                wv,
                wo,
                ffnNorm,
                wGate,
                wDown,
                wUp,
                rope);
    }

    private static Tensor mapTensorAsArrayTensor(
            GGUF gguf, String name, long tensorBase, FileChannel channel, Arena arena) {
        TensorEntry e = gguf.getTensor(name);
        if (e == null) {
            throw new IllegalArgumentException("Missing tensor: " + name);
        }
        if (e.ggmlType() != GGMLType.F32) {
            throw new UnsupportedOperationException(
                    "Only F32 is supported; " + name + " is " + e.ggmlType());
        }
        long bytes = e.ggmlType().byteSizeForShape(e.shape());
        try {
            MemorySegment seg =
                    channel.map(
                            FileChannel.MapMode.READ_ONLY, tensorBase + e.offset(), bytes, arena);
            Memory<MemorySegment> mem = MemoryFactory.ofMemorySegment(seg);
            Shape shape =
                    e.shape().length == 2
                            ? Shape.of(e.shape()[1], e.shape()[0])
                            : Shape.flat(e.shape());
            MemoryView<MemorySegment> view =
                    MemoryView.of(mem, 0, DataType.FP32, Layout.rowMajor(shape));
            int size = Math.toIntExact(shape.size());
            float[] out = new float[size];
            MemoryAccess<MemorySegment> access =
                    Environment.nativeMemoryDomain().directAccess();
            for (int i = 0; i < size; i++) {
                out[i] = access.readFloat(mem, i * 4L);
            }
            return Tensor.of(out, shape);
        } catch (IOException ex) {
            throw new RuntimeException("Failed to map tensor: " + name, ex);
        }
    }

    private static float[] toFloatArray(Tensor tensor) {
        MemoryView<?> view = tensor.materialize();
        MemoryDomain<MemorySegment> nativeDomain = Environment.nativeMemoryDomain();
        // If the tensor is on a device (e.g. HIP GPU), copy it to host memory first.
        if (!view.memory().device().belongsTo(DeviceType.PANAMA)) {
            MemoryView<MemorySegment> hostView =
                    MemoryView.of(
                            nativeDomain
                                    .memoryAllocator()
                                    .allocateMemory(view.dataType(), view.shape()),
                            view.dataType(),
                            Layout.rowMajor(view.shape()));
            @SuppressWarnings("unchecked")
            MemoryDomain<Object> srcDomain =
                    (MemoryDomain<Object>)
                            Environment.runtimeFor(view.memory().device()).memoryDomain();
            @SuppressWarnings("unchecked")
            MemoryView<Object> srcView = (MemoryView<Object>) view;
            MemoryDomain.copy(srcDomain, srcView, nativeDomain, hostView);
            view = hostView;
        }
        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment> host = (MemoryView<MemorySegment>) view;
        MemoryAccess<MemorySegment> access = nativeDomain.directAccess();
        int size = Math.toIntExact(host.shape().size());
        float[] out = new float[size];
        for (int i = 0; i < size; i++) {
            long off = Indexing.linearToOffset(host, i);
            out[i] = access.readFloat(host.memory(), off);
        }
        return out;
    }

    private record LlamaConfig(
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

    private record LlamaWeights(
            Tensor tokenTable,
            Tensor output,
            Tensor outputNorm,
            Tensor[] attnNorm,
            Tensor[] wq,
            Tensor[] wk,
            Tensor[] wv,
            Tensor[] wo,
            Tensor[] ffnNorm,
            Tensor[] wGate,
            Tensor[] wDown,
            Tensor[] wUp,
            RopeTables rope) {}

    private static final class Llama32Model {
        private static final String GEMV_KERNEL_NAME = "llama.gemv.fp32";
        private static final String GEMV2_KERNEL_NAME = "llama.gemv2.fp32";
        private static final String GEMV3_KERNEL_NAME = "llama.gemv3.fp32";
        private static final String GEMV_DOWN_KERNEL_NAME = "llama.gemv.down.fp32";
        private static final String SWIGLU_KERNEL_NAME = "llama.swiglu.fp32";
        private static final String C_GEMV_PAIR_KERNEL_NAME = "llama.gemv.pair.avx2.c";
        private static final String C_GEMV_DOWN_KERNEL_NAME = "llama.gemv.down.avx2.c";
        private static final String C_GEMV_LOGITS_KERNEL_NAME = "llama.gemv.logits.avx2.c";
        // language=java
        private static final String GEMV_KERNEL_SOURCE =
                """
                package com.qxotic.jota.tensor.jit;

                import java.util.stream.IntStream;
                import com.qxotic.jota.memory.MemoryAccess;
                import com.qxotic.jota.memory.MemoryDomain;
                import com.qxotic.jota.memory.MemoryView;
                import com.qxotic.jota.runtime.JavaKernel;
                import com.qxotic.jota.runtime.KernelArgs;
                import java.lang.foreign.MemorySegment;
                import java.nio.ByteOrder;
                import jdk.incubator.vector.FloatVector;
                import jdk.incubator.vector.VectorOperators;
                import jdk.incubator.vector.VectorSpecies;

                public final class LlamaGemvKernel implements JavaKernel {
                    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
                    private static final long PARALLEL_MIN_WORK = 131072L;
                    private static final int THREADS =
                            Math.max(
                                    1,
                                    Integer.getInteger(
                                            "com.qxotic.llama.kernel.threads",
                                            Runtime.getRuntime().availableProcessors()));
                    private static final java.util.concurrent.ExecutorService EXEC =
                            java.util.concurrent.Executors.newFixedThreadPool(
                                    THREADS,
                                    r -> {
                                        Thread t = new Thread(r, "llama-gemv");
                                        t.setDaemon(true);
                                        return t;
                                    });

                    @Override
                    @SuppressWarnings("unchecked")
                    public void execute(MemoryDomain<MemorySegment> domain, KernelArgs args) {
                        MemoryView<MemorySegment> a = (MemoryView<MemorySegment>) args.getBuffer(0);
                        MemoryView<MemorySegment> x = (MemoryView<MemorySegment>) args.getBuffer(1);
                        MemoryView<MemorySegment> y = (MemoryView<MemorySegment>) args.getBuffer(2);
                        int m = args.getInt(3);
                        int n = args.getInt(4);

                        MemoryAccess<MemorySegment> access = domain.directAccess();
                        MemorySegment aBaseSegment = a.memory().base();
                        MemorySegment xBaseSegment = x.memory().base();
                        long aBase = a.byteOffset();
                        long xBase = x.byteOffset();
                        long yBase = y.byteOffset();
                        long work = (long) m * n;
                        if (work < PARALLEL_MIN_WORK) {
                            for (int row = 0; row < m; row++) {
                                long rowBase = aBase + (long) row * n * Float.BYTES;
                                int col = 0;
                                FloatVector acc = FloatVector.zero(SPECIES);
                                int upper = SPECIES.loopBound(n);
                                for (; col < upper; col += SPECIES.length()) {
                                    long aOff = rowBase + (long) col * Float.BYTES;
                                    long xOff = xBase + (long) col * Float.BYTES;
                                    FloatVector xv =
                                            FloatVector.fromMemorySegment(
                                                    SPECIES,
                                                    xBaseSegment,
                                                    xOff,
                                                    ByteOrder.nativeOrder());
                                    FloatVector av =
                                            FloatVector.fromMemorySegment(
                                                    SPECIES,
                                                    aBaseSegment,
                                                    aOff,
                                                    ByteOrder.nativeOrder());
                                    acc = av.fma(xv, acc);
                                }
                                float dot = acc.reduceLanes(VectorOperators.ADD);
                                for (; col < n; col++) {
                                    float av = access.readFloat(a.memory(), rowBase + (long) col * Float.BYTES);
                                    float xv = access.readFloat(x.memory(), xBase + (long) col * Float.BYTES);
                                    dot += av * xv;
                                }
                                access.writeFloat(y.memory(), yBase + (long) row * Float.BYTES, dot);
                            }
                            return;
                        }

                        parallelForRows(m, row -> {
                            long rowBase = aBase + (long) row * n * Float.BYTES;
                            int col = 0;
                            FloatVector acc = FloatVector.zero(SPECIES);
                            int upper = SPECIES.loopBound(n);
                            for (; col < upper; col += SPECIES.length()) {
                                long aOff = rowBase + (long) col * Float.BYTES;
                                long xOff = xBase + (long) col * Float.BYTES;
                                FloatVector xv =
                                        FloatVector.fromMemorySegment(
                                                SPECIES,
                                                xBaseSegment,
                                                xOff,
                                                ByteOrder.nativeOrder());
                                FloatVector av =
                                        FloatVector.fromMemorySegment(
                                                SPECIES,
                                                aBaseSegment,
                                                aOff,
                                                ByteOrder.nativeOrder());
                                acc = av.fma(xv, acc);
                            }
                            float dot = acc.reduceLanes(VectorOperators.ADD);
                            for (; col < n; col++) {
                                float av = access.readFloat(a.memory(), rowBase + (long) col * Float.BYTES);
                                float xv = access.readFloat(x.memory(), xBase + (long) col * Float.BYTES);
                                dot += av * xv;
                            }
                            access.writeFloat(y.memory(), yBase + (long) row * Float.BYTES, dot);
                        });
                    }

                    private static void parallelForRows(int rows, java.util.function.IntConsumer body) {
                        int workers = Math.min(THREADS, rows);
                        if (workers <= 1) {
                            for (int row = 0; row < rows; row++) {
                                body.accept(row);
                            }
                            return;
                        }
                        int chunk = (rows + workers - 1) / workers;
                        java.util.List<java.util.concurrent.Future<?>> futures =
                                new java.util.ArrayList<>(workers);
                        for (int w = 0; w < workers; w++) {
                            int start = w * chunk;
                            int end = Math.min(rows, start + chunk);
                            if (start >= end) {
                                continue;
                            }
                            futures.add(
                                    EXEC.submit(
                                            () -> {
                                                for (int row = start; row < end; row++) {
                                                    body.accept(row);
                                                }
                                            }));
                        }
                        for (java.util.concurrent.Future<?> future : futures) {
                            try {
                                future.get();
                            } catch (InterruptedException ex) {
                                Thread.currentThread().interrupt();
                                throw new RuntimeException("GEMV worker interrupted", ex);
                            } catch (java.util.concurrent.ExecutionException ex) {
                                throw new RuntimeException("GEMV worker failed", ex.getCause());
                            }
                        }
                    }
                }
                """;
        // language=java
        private static final String GEMV2_KERNEL_SOURCE =
                """
                package com.qxotic.jota.tensor.jit;

                import java.util.stream.IntStream;
                import com.qxotic.jota.memory.MemoryAccess;
                import com.qxotic.jota.memory.MemoryDomain;
                import com.qxotic.jota.memory.MemoryView;
                import com.qxotic.jota.runtime.JavaKernel;
                import com.qxotic.jota.runtime.KernelArgs;
                import java.lang.foreign.MemorySegment;
                import java.nio.ByteOrder;
                import jdk.incubator.vector.FloatVector;
                import jdk.incubator.vector.VectorOperators;
                import jdk.incubator.vector.VectorSpecies;

                public final class LlamaGemv2Kernel implements JavaKernel {
                    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
                    private static final long PARALLEL_MIN_WORK = 131072L;
                    private static final int THREADS =
                            Math.max(
                                    1,
                                    Integer.getInteger(
                                            "com.qxotic.llama.kernel.threads",
                                            Runtime.getRuntime().availableProcessors()));
                    private static final java.util.concurrent.ExecutorService EXEC =
                            java.util.concurrent.Executors.newFixedThreadPool(
                                    THREADS,
                                    r -> {
                                        Thread t = new Thread(r, "llama-gemv2");
                                        t.setDaemon(true);
                                        return t;
                                    });

                    @Override
                    @SuppressWarnings("unchecked")
                    public void execute(MemoryDomain<MemorySegment> domain, KernelArgs args) {
                        MemoryView<MemorySegment> a0 = (MemoryView<MemorySegment>) args.getBuffer(0);
                        MemoryView<MemorySegment> a1 = (MemoryView<MemorySegment>) args.getBuffer(1);
                        MemoryView<MemorySegment> x = (MemoryView<MemorySegment>) args.getBuffer(2);
                        MemoryView<MemorySegment> out = (MemoryView<MemorySegment>) args.getBuffer(3);
                        int m = args.getInt(4);
                        int n = args.getInt(5);

                        MemoryAccess<MemorySegment> access = domain.directAccess();
                        MemorySegment a0BaseSegment = a0.memory().base();
                        MemorySegment a1BaseSegment = a1.memory().base();
                        MemorySegment xBaseSegment = x.memory().base();
                        long a0Base = a0.byteOffset();
                        long a1Base = a1.byteOffset();
                        long xBase = x.byteOffset();
                        long outBase = out.byteOffset();

                        long work = (long) m * n;
                        if (work < PARALLEL_MIN_WORK) {
                            for (int row = 0; row < m; row++) {
                                long row0Base = a0Base + (long) row * n * Float.BYTES;
                                long row1Base = a1Base + (long) row * n * Float.BYTES;
                                int col = 0;
                                FloatVector acc0 = FloatVector.zero(SPECIES);
                                FloatVector acc1 = FloatVector.zero(SPECIES);
                                int upper = SPECIES.loopBound(n);
                                for (; col < upper; col += SPECIES.length()) {
                                    long xOff = xBase + (long) col * Float.BYTES;
                                    FloatVector xv =
                                            FloatVector.fromMemorySegment(
                                                    SPECIES,
                                                    xBaseSegment,
                                                    xOff,
                                                    ByteOrder.nativeOrder());
                                    long a0Off = row0Base + (long) col * Float.BYTES;
                                    long a1Off = row1Base + (long) col * Float.BYTES;
                                    FloatVector av0 =
                                            FloatVector.fromMemorySegment(
                                                    SPECIES,
                                                    a0BaseSegment,
                                                    a0Off,
                                                    ByteOrder.nativeOrder());
                                    FloatVector av1 =
                                            FloatVector.fromMemorySegment(
                                                    SPECIES,
                                                    a1BaseSegment,
                                                    a1Off,
                                                    ByteOrder.nativeOrder());
                                    acc0 = av0.fma(xv, acc0);
                                    acc1 = av1.fma(xv, acc1);
                                }
                                float dot0 = acc0.reduceLanes(VectorOperators.ADD);
                                float dot1 = acc1.reduceLanes(VectorOperators.ADD);
                                for (; col < n; col++) {
                                    float xv = access.readFloat(x.memory(), xBase + (long) col * Float.BYTES);
                                    float av0 = access.readFloat(a0.memory(), row0Base + (long) col * Float.BYTES);
                                    float av1 = access.readFloat(a1.memory(), row1Base + (long) col * Float.BYTES);
                                    dot0 += av0 * xv;
                                    dot1 += av1 * xv;
                                }
                                float swiglu = dot0 / (1.0f + (float) Math.exp(-dot0));
                                access.writeFloat(out.memory(), outBase + (long) row * Float.BYTES, swiglu * dot1);
                            }
                            return;
                        }

                        parallelForRows(m, row -> {
                            long row0Base = a0Base + (long) row * n * Float.BYTES;
                            long row1Base = a1Base + (long) row * n * Float.BYTES;
                            int col = 0;
                            FloatVector acc0 = FloatVector.zero(SPECIES);
                            FloatVector acc1 = FloatVector.zero(SPECIES);
                            int upper = SPECIES.loopBound(n);
                            for (; col < upper; col += SPECIES.length()) {
                                long xOff = xBase + (long) col * Float.BYTES;
                                FloatVector xv =
                                        FloatVector.fromMemorySegment(
                                                SPECIES,
                                                xBaseSegment,
                                                xOff,
                                                ByteOrder.nativeOrder());
                                long a0Off = row0Base + (long) col * Float.BYTES;
                                long a1Off = row1Base + (long) col * Float.BYTES;
                                FloatVector av0 =
                                        FloatVector.fromMemorySegment(
                                                SPECIES,
                                                a0BaseSegment,
                                                a0Off,
                                                ByteOrder.nativeOrder());
                                FloatVector av1 =
                                        FloatVector.fromMemorySegment(
                                                SPECIES,
                                                a1BaseSegment,
                                                a1Off,
                                                ByteOrder.nativeOrder());
                                acc0 = av0.fma(xv, acc0);
                                acc1 = av1.fma(xv, acc1);
                            }
                            float dot0 = acc0.reduceLanes(VectorOperators.ADD);
                            float dot1 = acc1.reduceLanes(VectorOperators.ADD);
                            for (; col < n; col++) {
                                float xv = access.readFloat(x.memory(), xBase + (long) col * Float.BYTES);
                                float av0 = access.readFloat(a0.memory(), row0Base + (long) col * Float.BYTES);
                                float av1 = access.readFloat(a1.memory(), row1Base + (long) col * Float.BYTES);
                                dot0 += av0 * xv;
                                dot1 += av1 * xv;
                            }
                            float swiglu = dot0 / (1.0f + (float) Math.exp(-dot0));
                            access.writeFloat(out.memory(), outBase + (long) row * Float.BYTES, swiglu * dot1);
                        });
                    }

                    private static void parallelForRows(int rows, java.util.function.IntConsumer body) {
                        int workers = Math.min(THREADS, rows);
                        if (workers <= 1) {
                            for (int row = 0; row < rows; row++) {
                                body.accept(row);
                            }
                            return;
                        }
                        int chunk = (rows + workers - 1) / workers;
                        java.util.List<java.util.concurrent.Future<?>> futures =
                                new java.util.ArrayList<>(workers);
                        for (int w = 0; w < workers; w++) {
                            int start = w * chunk;
                            int end = Math.min(rows, start + chunk);
                            if (start >= end) {
                                continue;
                            }
                            futures.add(
                                    EXEC.submit(
                                            () -> {
                                                for (int row = start; row < end; row++) {
                                                    body.accept(row);
                                                }
                                            }));
                        }
                        for (java.util.concurrent.Future<?> future : futures) {
                            try {
                                future.get();
                            } catch (InterruptedException ex) {
                                Thread.currentThread().interrupt();
                                throw new RuntimeException("GEMV2 worker interrupted", ex);
                            } catch (java.util.concurrent.ExecutionException ex) {
                                throw new RuntimeException("GEMV2 worker failed", ex.getCause());
                            }
                        }
                    }
                }
                """;
        // language=java
        private static final String GEMV3_KERNEL_SOURCE =
                """
                package com.qxotic.jota.tensor.jit;

                import java.util.stream.IntStream;
                import com.qxotic.jota.memory.MemoryAccess;
                import com.qxotic.jota.memory.MemoryDomain;
                import com.qxotic.jota.memory.MemoryView;
                import com.qxotic.jota.runtime.JavaKernel;
                import com.qxotic.jota.runtime.KernelArgs;
                import java.lang.foreign.MemorySegment;
                import java.nio.ByteOrder;
                import jdk.incubator.vector.FloatVector;
                import jdk.incubator.vector.VectorOperators;
                import jdk.incubator.vector.VectorSpecies;

                public final class LlamaGemv3Kernel implements JavaKernel {
                    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
                    private static final long PARALLEL_MIN_WORK = 131072L;
                    private static final int THREADS =
                            Math.max(
                                    1,
                                    Integer.getInteger(
                                            "com.qxotic.llama.kernel.threads",
                                            Runtime.getRuntime().availableProcessors()));
                    private static final java.util.concurrent.ExecutorService EXEC =
                            java.util.concurrent.Executors.newFixedThreadPool(
                                    THREADS,
                                    r -> {
                                        Thread t = new Thread(r, "llama-gemv3");
                                        t.setDaemon(true);
                                        return t;
                                    });

                    @Override
                    @SuppressWarnings("unchecked")
                    public void execute(MemoryDomain<MemorySegment> domain, KernelArgs args) {
                        MemoryView<MemorySegment> wq = (MemoryView<MemorySegment>) args.getBuffer(0);
                        MemoryView<MemorySegment> wk = (MemoryView<MemorySegment>) args.getBuffer(1);
                        MemoryView<MemorySegment> wv = (MemoryView<MemorySegment>) args.getBuffer(2);
                        MemoryView<MemorySegment> x = (MemoryView<MemorySegment>) args.getBuffer(3);
                        MemoryView<MemorySegment> outQ = (MemoryView<MemorySegment>) args.getBuffer(4);
                        MemoryView<MemorySegment> outK = (MemoryView<MemorySegment>) args.getBuffer(5);
                        MemoryView<MemorySegment> outV = (MemoryView<MemorySegment>) args.getBuffer(6);
                        int mQ = args.getInt(7);
                        int mKV = args.getInt(8);
                        int n = args.getInt(9);

                        MemoryAccess<MemorySegment> access = domain.directAccess();
                        MemorySegment wqSeg = wq.memory().base();
                        MemorySegment wkSeg = wk.memory().base();
                        MemorySegment wvSeg = wv.memory().base();
                        MemorySegment xSeg = x.memory().base();
                        long wqBase = wq.byteOffset();
                        long wkBase = wk.byteOffset();
                        long wvBase = wv.byteOffset();
                        long xBase = x.byteOffset();
                        long outQBase = outQ.byteOffset();
                        long outKBase = outK.byteOffset();
                        long outVBase = outV.byteOffset();

                        long work = (long) (mQ + 2L * mKV) * n;
                        if (work < PARALLEL_MIN_WORK) {
                            for (int row = 0; row < mQ; row++) {
                                long rowBase = wqBase + (long) row * n * Float.BYTES;
                                int col = 0;
                                FloatVector acc = FloatVector.zero(SPECIES);
                                int upper = SPECIES.loopBound(n);
                                for (; col < upper; col += SPECIES.length()) {
                                    long xOff = xBase + (long) col * Float.BYTES;
                                    long wOff = rowBase + (long) col * Float.BYTES;
                                    FloatVector xv = FloatVector.fromMemorySegment(SPECIES, xSeg, xOff, ByteOrder.nativeOrder());
                                    FloatVector wv0 = FloatVector.fromMemorySegment(SPECIES, wqSeg, wOff, ByteOrder.nativeOrder());
                                    acc = wv0.fma(xv, acc);
                                }
                                float dot = acc.reduceLanes(VectorOperators.ADD);
                                for (; col < n; col++) {
                                    float xv = access.readFloat(x.memory(), xBase + (long) col * Float.BYTES);
                                    float qv = access.readFloat(wq.memory(), rowBase + (long) col * Float.BYTES);
                                    dot += qv * xv;
                                }
                                access.writeFloat(outQ.memory(), outQBase + (long) row * Float.BYTES, dot);
                            }
                            for (int row = 0; row < mKV; row++) {
                                long rowKBase = wkBase + (long) row * n * Float.BYTES;
                                long rowVBase = wvBase + (long) row * n * Float.BYTES;
                                int col = 0;
                                FloatVector accK = FloatVector.zero(SPECIES);
                                FloatVector accV = FloatVector.zero(SPECIES);
                                int upper = SPECIES.loopBound(n);
                                for (; col < upper; col += SPECIES.length()) {
                                    long xOff = xBase + (long) col * Float.BYTES;
                                    long kOff = rowKBase + (long) col * Float.BYTES;
                                    long vOff = rowVBase + (long) col * Float.BYTES;
                                    FloatVector xv = FloatVector.fromMemorySegment(SPECIES, xSeg, xOff, ByteOrder.nativeOrder());
                                    FloatVector kv = FloatVector.fromMemorySegment(SPECIES, wkSeg, kOff, ByteOrder.nativeOrder());
                                    FloatVector vv = FloatVector.fromMemorySegment(SPECIES, wvSeg, vOff, ByteOrder.nativeOrder());
                                    accK = kv.fma(xv, accK);
                                    accV = vv.fma(xv, accV);
                                }
                                float dotK = accK.reduceLanes(VectorOperators.ADD);
                                float dotV = accV.reduceLanes(VectorOperators.ADD);
                                for (; col < n; col++) {
                                    float xv = access.readFloat(x.memory(), xBase + (long) col * Float.BYTES);
                                    float kv = access.readFloat(wk.memory(), rowKBase + (long) col * Float.BYTES);
                                    float vv = access.readFloat(wv.memory(), rowVBase + (long) col * Float.BYTES);
                                    dotK += kv * xv;
                                    dotV += vv * xv;
                                }
                                access.writeFloat(outK.memory(), outKBase + (long) row * Float.BYTES, dotK);
                                access.writeFloat(outV.memory(), outVBase + (long) row * Float.BYTES, dotV);
                            }
                            return;
                        }

                        parallelForRows(mQ, row -> {
                            long rowBase = wqBase + (long) row * n * Float.BYTES;
                            int col = 0;
                            FloatVector acc = FloatVector.zero(SPECIES);
                            int upper = SPECIES.loopBound(n);
                            for (; col < upper; col += SPECIES.length()) {
                                long xOff = xBase + (long) col * Float.BYTES;
                                long wOff = rowBase + (long) col * Float.BYTES;
                                FloatVector xv = FloatVector.fromMemorySegment(SPECIES, xSeg, xOff, ByteOrder.nativeOrder());
                                FloatVector qv = FloatVector.fromMemorySegment(SPECIES, wqSeg, wOff, ByteOrder.nativeOrder());
                                acc = qv.fma(xv, acc);
                            }
                            float dot = acc.reduceLanes(VectorOperators.ADD);
                            for (; col < n; col++) {
                                float xv = access.readFloat(x.memory(), xBase + (long) col * Float.BYTES);
                                float qv = access.readFloat(wq.memory(), rowBase + (long) col * Float.BYTES);
                                dot += qv * xv;
                            }
                            access.writeFloat(outQ.memory(), outQBase + (long) row * Float.BYTES, dot);
                        });
                        parallelForRows(mKV, row -> {
                            long rowKBase = wkBase + (long) row * n * Float.BYTES;
                            long rowVBase = wvBase + (long) row * n * Float.BYTES;
                            int col = 0;
                            FloatVector accK = FloatVector.zero(SPECIES);
                            FloatVector accV = FloatVector.zero(SPECIES);
                            int upper = SPECIES.loopBound(n);
                            for (; col < upper; col += SPECIES.length()) {
                                long xOff = xBase + (long) col * Float.BYTES;
                                long kOff = rowKBase + (long) col * Float.BYTES;
                                long vOff = rowVBase + (long) col * Float.BYTES;
                                FloatVector xv = FloatVector.fromMemorySegment(SPECIES, xSeg, xOff, ByteOrder.nativeOrder());
                                FloatVector kv = FloatVector.fromMemorySegment(SPECIES, wkSeg, kOff, ByteOrder.nativeOrder());
                                FloatVector vv = FloatVector.fromMemorySegment(SPECIES, wvSeg, vOff, ByteOrder.nativeOrder());
                                accK = kv.fma(xv, accK);
                                accV = vv.fma(xv, accV);
                            }
                            float dotK = accK.reduceLanes(VectorOperators.ADD);
                            float dotV = accV.reduceLanes(VectorOperators.ADD);
                            for (; col < n; col++) {
                                float xv = access.readFloat(x.memory(), xBase + (long) col * Float.BYTES);
                                float kv = access.readFloat(wk.memory(), rowKBase + (long) col * Float.BYTES);
                                float vv = access.readFloat(wv.memory(), rowVBase + (long) col * Float.BYTES);
                                dotK += kv * xv;
                                dotV += vv * xv;
                            }
                            access.writeFloat(outK.memory(), outKBase + (long) row * Float.BYTES, dotK);
                            access.writeFloat(outV.memory(), outVBase + (long) row * Float.BYTES, dotV);
                        });
                    }

                    private static void parallelForRows(int rows, java.util.function.IntConsumer body) {
                        int workers = Math.min(THREADS, rows);
                        if (workers <= 1) {
                            for (int row = 0; row < rows; row++) {
                                body.accept(row);
                            }
                            return;
                        }
                        int chunk = (rows + workers - 1) / workers;
                        java.util.List<java.util.concurrent.Future<?>> futures =
                                new java.util.ArrayList<>(workers);
                        for (int w = 0; w < workers; w++) {
                            int start = w * chunk;
                            int end = Math.min(rows, start + chunk);
                            if (start >= end) {
                                continue;
                            }
                            futures.add(
                                    EXEC.submit(
                                            () -> {
                                                for (int row = start; row < end; row++) {
                                                    body.accept(row);
                                                }
                                            }));
                        }
                        for (java.util.concurrent.Future<?> future : futures) {
                            try {
                                future.get();
                            } catch (InterruptedException ex) {
                                Thread.currentThread().interrupt();
                                throw new RuntimeException("GEMV3 worker interrupted", ex);
                            } catch (java.util.concurrent.ExecutionException ex) {
                                throw new RuntimeException("GEMV3 worker failed", ex.getCause());
                            }
                        }
                    }
                }
                """;
        // language=java
        private static final String GEMV_DOWN_KERNEL_SOURCE =
                """
                package com.qxotic.jota.tensor.jit;

                import java.util.stream.IntStream;
                import com.qxotic.jota.memory.MemoryAccess;
                import com.qxotic.jota.memory.MemoryDomain;
                import com.qxotic.jota.memory.MemoryView;
                import com.qxotic.jota.runtime.JavaKernel;
                import com.qxotic.jota.runtime.KernelArgs;
                import java.lang.foreign.MemorySegment;
                import java.nio.ByteOrder;
                import jdk.incubator.vector.FloatVector;
                import jdk.incubator.vector.VectorOperators;
                import jdk.incubator.vector.VectorSpecies;

                public final class LlamaGemvDownKernel implements JavaKernel {
                    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
                    private static final long PARALLEL_MIN_WORK = 196608L;
                    private static final int THREADS =
                            Math.max(
                                    1,
                                    Integer.getInteger(
                                            "com.qxotic.llama.kernel.threads",
                                            Runtime.getRuntime().availableProcessors()));
                    private static final java.util.concurrent.ExecutorService EXEC =
                            java.util.concurrent.Executors.newFixedThreadPool(
                                    THREADS,
                                    r -> {
                                        Thread t = new Thread(r, "llama-gemv-down");
                                        t.setDaemon(true);
                                        return t;
                                    });

                    @Override
                    @SuppressWarnings("unchecked")
                    public void execute(MemoryDomain<MemorySegment> domain, KernelArgs args) {
                        MemoryView<MemorySegment> a = (MemoryView<MemorySegment>) args.getBuffer(0);
                        MemoryView<MemorySegment> x = (MemoryView<MemorySegment>) args.getBuffer(1);
                        MemoryView<MemorySegment> y = (MemoryView<MemorySegment>) args.getBuffer(2);
                        int m = args.getInt(3);
                        int n = args.getInt(4);

                        MemoryAccess<MemorySegment> access = domain.directAccess();
                        MemorySegment aSeg = a.memory().base();
                        MemorySegment xSeg = x.memory().base();
                        long aBase = a.byteOffset();
                        long xBase = x.byteOffset();
                        long yBase = y.byteOffset();
                        int lanes = SPECIES.length();

                        java.util.function.IntConsumer body = row -> {
                            long rowBase = aBase + (long) row * n * Float.BYTES;
                            int col = 0;
                            FloatVector acc0 = FloatVector.zero(SPECIES);
                            FloatVector acc1 = FloatVector.zero(SPECIES);
                            int upper = SPECIES.loopBound(n);
                            int upper2 = upper - (upper % (2 * lanes));
                            for (; col < upper2; col += 2 * lanes) {
                                long aOff0 = rowBase + (long) col * Float.BYTES;
                                long xOff0 = xBase + (long) col * Float.BYTES;
                                long aOff1 = aOff0 + (long) lanes * Float.BYTES;
                                long xOff1 = xOff0 + (long) lanes * Float.BYTES;
                                FloatVector av0 = FloatVector.fromMemorySegment(SPECIES, aSeg, aOff0, ByteOrder.nativeOrder());
                                FloatVector xv0 = FloatVector.fromMemorySegment(SPECIES, xSeg, xOff0, ByteOrder.nativeOrder());
                                FloatVector av1 = FloatVector.fromMemorySegment(SPECIES, aSeg, aOff1, ByteOrder.nativeOrder());
                                FloatVector xv1 = FloatVector.fromMemorySegment(SPECIES, xSeg, xOff1, ByteOrder.nativeOrder());
                                acc0 = av0.fma(xv0, acc0);
                                acc1 = av1.fma(xv1, acc1);
                            }
                            FloatVector acc = acc0.add(acc1);
                            for (; col < upper; col += lanes) {
                                long aOff = rowBase + (long) col * Float.BYTES;
                                long xOff = xBase + (long) col * Float.BYTES;
                                FloatVector av = FloatVector.fromMemorySegment(SPECIES, aSeg, aOff, ByteOrder.nativeOrder());
                                FloatVector xv = FloatVector.fromMemorySegment(SPECIES, xSeg, xOff, ByteOrder.nativeOrder());
                                acc = av.fma(xv, acc);
                            }
                            float dot = acc.reduceLanes(VectorOperators.ADD);
                            for (; col < n; col++) {
                                float av = access.readFloat(a.memory(), rowBase + (long) col * Float.BYTES);
                                float xv = access.readFloat(x.memory(), xBase + (long) col * Float.BYTES);
                                dot += av * xv;
                            }
                            access.writeFloat(y.memory(), yBase + (long) row * Float.BYTES, dot);
                        };

                        long work = (long) m * n;
                        if (work < PARALLEL_MIN_WORK) {
                            for (int row = 0; row < m; row++) {
                                body.accept(row);
                            }
                            return;
                        }
                        parallelForRows(m, body);
                    }

                    private static void parallelForRows(int rows, java.util.function.IntConsumer body) {
                        int workers = Math.min(THREADS, rows);
                        if (workers <= 1) {
                            for (int row = 0; row < rows; row++) {
                                body.accept(row);
                            }
                            return;
                        }
                        int chunk = (rows + workers - 1) / workers;
                        java.util.List<java.util.concurrent.Future<?>> futures =
                                new java.util.ArrayList<>(workers);
                        for (int w = 0; w < workers; w++) {
                            int start = w * chunk;
                            int end = Math.min(rows, start + chunk);
                            if (start >= end) {
                                continue;
                            }
                            futures.add(
                                    EXEC.submit(
                                            () -> {
                                                for (int row = start; row < end; row++) {
                                                    body.accept(row);
                                                }
                                            }));
                        }
                        for (java.util.concurrent.Future<?> future : futures) {
                            try {
                                future.get();
                            } catch (InterruptedException ex) {
                                Thread.currentThread().interrupt();
                                throw new RuntimeException("GEMV-down worker interrupted", ex);
                            } catch (java.util.concurrent.ExecutionException ex) {
                                throw new RuntimeException(
                                        "GEMV-down worker failed", ex.getCause());
                            }
                        }
                    }
                }
                """;
        // language=java
        private static final String SWIGLU_KERNEL_SOURCE =
                """
                package com.qxotic.jota.tensor.jit;

                import java.util.stream.IntStream;
                import com.qxotic.jota.memory.MemoryAccess;
                import com.qxotic.jota.memory.MemoryDomain;
                import com.qxotic.jota.memory.MemoryView;
                import com.qxotic.jota.runtime.JavaKernel;
                import com.qxotic.jota.runtime.KernelArgs;
                import java.lang.foreign.MemorySegment;
                import java.nio.ByteOrder;
                import jdk.incubator.vector.FloatVector;
                import jdk.incubator.vector.VectorSpecies;

                public final class LlamaSwigluKernel implements JavaKernel {
                    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
                    private static final long PARALLEL_MIN_WORK = 32768L;

                    @Override
                    @SuppressWarnings("unchecked")
                    public void execute(MemoryDomain<MemorySegment> domain, KernelArgs args) {
                        MemoryView<MemorySegment> gate = (MemoryView<MemorySegment>) args.getBuffer(0);
                        MemoryView<MemorySegment> up = (MemoryView<MemorySegment>) args.getBuffer(1);
                        MemoryView<MemorySegment> out = (MemoryView<MemorySegment>) args.getBuffer(2);
                        int n = args.getInt(3);

                        MemoryAccess<MemorySegment> access = domain.directAccess();
                        MemorySegment gateSegment = gate.memory().base();
                        MemorySegment upSegment = up.memory().base();
                        MemorySegment outSegment = out.memory().base();
                        long gateBase = gate.byteOffset();
                        long upBase = up.byteOffset();
                        long outBase = out.byteOffset();

                        if (n < PARALLEL_MIN_WORK) {
                            int i = 0;
                            int upper = SPECIES.loopBound(n);
                            for (; i < upper; i += SPECIES.length()) {
                                long gateOff = gateBase + (long) i * Float.BYTES;
                                long upOff = upBase + (long) i * Float.BYTES;
                                long outOff = outBase + (long) i * Float.BYTES;
                                FloatVector gateV =
                                        FloatVector.fromMemorySegment(
                                                SPECIES, gateSegment, gateOff, ByteOrder.nativeOrder());
                                FloatVector upV =
                                        FloatVector.fromMemorySegment(
                                                SPECIES, upSegment, upOff, ByteOrder.nativeOrder());
                                FloatVector one = FloatVector.broadcast(SPECIES, 1f);
                                FloatVector silu = gateV.div(gateV.neg().lanewise(jdk.incubator.vector.VectorOperators.EXP).add(one));
                                silu.mul(upV)
                                        .intoMemorySegment(
                                                outSegment,
                                                outOff,
                                                ByteOrder.nativeOrder());
                            }
                            for (; i < n; i++) {
                                float g = access.readFloat(gate.memory(), gateBase + (long) i * Float.BYTES);
                                float u = access.readFloat(up.memory(), upBase + (long) i * Float.BYTES);
                                float silu = g / (1.0f + (float) Math.exp(-g));
                                access.writeFloat(out.memory(), outBase + (long) i * Float.BYTES, silu * u);
                            }
                            return;
                        }

                        IntStream.range(0, n).parallel().forEach(i -> {
                            float g = access.readFloat(gate.memory(), gateBase + (long) i * Float.BYTES);
                            float u = access.readFloat(up.memory(), upBase + (long) i * Float.BYTES);
                            float silu = g / (1.0f + (float) Math.exp(-g));
                            access.writeFloat(out.memory(), outBase + (long) i * Float.BYTES, silu * u);
                        });
                    }
                }
                """;
        private static final int C_GEMV_OMP_MIN_ROWS_DOWN =
                Integer.getInteger("com.qxotic.llama.cgemv.omp.minRows.down", 64);
        private static final int C_GEMV_OMP_MIN_ROWS_LOGITS =
                Integer.getInteger("com.qxotic.llama.cgemv.omp.minRows.logits", 32);
        private static final int C_GEMV_OMP_MIN_ROWS_PAIR =
                Integer.getInteger("com.qxotic.llama.cgemv.omp.minRows.pair", 64);
        private static final int C_GEMV_OMP_CHUNK_DOWN =
                Integer.getInteger("com.qxotic.llama.cgemv.omp.chunk.down", 4);
        private static final int C_GEMV_OMP_CHUNK_LOGITS =
                Integer.getInteger("com.qxotic.llama.cgemv.omp.chunk.logits", 4);
        private static final int C_GEMV_OMP_CHUNK_PAIR =
                Integer.getInteger("com.qxotic.llama.cgemv.omp.chunk.pair", 4);
        private static final int C_GEMV_PREFETCH_FLOATS_DOWN =
                Integer.getInteger("com.qxotic.llama.cgemv.prefetch.down", 64);
        private static final int C_GEMV_PREFETCH_FLOATS_LOGITS =
                Integer.getInteger("com.qxotic.llama.cgemv.prefetch.logits", 64);
        private static final int C_GEMV_PREFETCH_FLOATS_PAIR =
                Integer.getInteger("com.qxotic.llama.cgemv.prefetch.pair", 128);
        private static final String C_GEMV_DOWN_KERNEL_SOURCE =
                buildCGemvKernelSource(
                        "llama_gemv_down_avx2",
                        C_GEMV_OMP_MIN_ROWS_DOWN,
                        C_GEMV_OMP_CHUNK_DOWN,
                        C_GEMV_PREFETCH_FLOATS_DOWN);
        private static final String C_GEMV_LOGITS_KERNEL_SOURCE =
                buildCGemvKernelSource(
                        "llama_gemv_logits_avx2",
                        C_GEMV_OMP_MIN_ROWS_LOGITS,
                        C_GEMV_OMP_CHUNK_LOGITS,
                        C_GEMV_PREFETCH_FLOATS_LOGITS);
        private static final String C_GEMV_PAIR_KERNEL_SOURCE =
                buildCGemvPairSwigluKernelSource(
                        "llama_gemv_pair_swiglu_avx2",
                        C_GEMV_OMP_MIN_ROWS_PAIR,
                        C_GEMV_OMP_CHUNK_PAIR,
                        C_GEMV_PREFETCH_FLOATS_PAIR);

        private static String buildCGemvKernelSource(
                String functionName, int ompMinRows, int ompChunk, int prefetchFloats) {
            int minRows = Math.max(1, ompMinRows);
            int chunk = Math.max(1, ompChunk);
            int prefetch = Math.max(16, prefetchFloats);
            return String.format(
                    Locale.ROOT,
                    """
                    #include <stdint.h>
                    #include <stddef.h>
                    #include <math.h>
                    #if defined(__x86_64__) || defined(__i386__)
                    #include <immintrin.h>
                    #endif

                    #define LLAMA_OMP_MIN_ROWS %d
                    #define LLAMA_OMP_CHUNK %d
                    #define LLAMA_PREFETCH_FLOATS %d

                    static inline void gemv_scalar(const float *A, const float *x, float *y, int M, int N) {
                    #if defined(_OPENMP)
                        #pragma omp parallel for schedule(static, LLAMA_OMP_CHUNK) if(M >= LLAMA_OMP_MIN_ROWS)
                    #endif
                        for (int row = 0; row < M; row++) {
                            float dot = 0.0f;
                            const float *a = A + (size_t) row * (size_t) N;
                            for (int col = 0; col < N; col++) {
                                dot += a[col] * x[col];
                            }
                            y[row] = dot;
                        }
                    }

                    #if defined(__x86_64__) || defined(__i386__)
                    static inline float hsum256_ps(__m256 v) {
                        __m128 lo = _mm256_castps256_ps128(v);
                        __m128 hi = _mm256_extractf128_ps(v, 1);
                        __m128 sum = _mm_add_ps(lo, hi);
                        __m128 shuf = _mm_movehdup_ps(sum);
                        __m128 sums = _mm_add_ps(sum, shuf);
                        shuf = _mm_movehl_ps(shuf, sums);
                        sums = _mm_add_ss(sums, shuf);
                        return _mm_cvtss_f32(sums);
                    }

                    #if defined(__GNUC__) || defined(__clang__)
                    __attribute__((target("avx2,fma")))
                    #endif
                    static void gemv_avx2_fma(const float *A, const float *x, float *y, int M, int N) {
                        const int lanes = 8;
                        const int unroll = 4;
                        const int step = lanes * unroll;
                        int upper = N - (N %% step);
                        int upperVec = N - (N %% lanes);
                    #if defined(_OPENMP)
                        #pragma omp parallel for schedule(static, LLAMA_OMP_CHUNK) if(M >= LLAMA_OMP_MIN_ROWS)
                    #endif
                        for (int row = 0; row < M; row++) {
                            const float *a = A + (size_t) row * (size_t) N;
                            __m256 acc0 = _mm256_setzero_ps();
                            __m256 acc1 = _mm256_setzero_ps();
                            __m256 acc2 = _mm256_setzero_ps();
                            __m256 acc3 = _mm256_setzero_ps();
                            int col = 0;
                            for (; col < upper; col += step) {
                    #if defined(__GNUC__) || defined(__clang__)
                                __builtin_prefetch(a + col + LLAMA_PREFETCH_FLOATS, 0, 1);
                                __builtin_prefetch(x + col + LLAMA_PREFETCH_FLOATS, 0, 1);
                    #endif
                                __m256 xv0 = _mm256_loadu_ps(x + col + 0 * lanes);
                                __m256 xv1 = _mm256_loadu_ps(x + col + 1 * lanes);
                                __m256 xv2 = _mm256_loadu_ps(x + col + 2 * lanes);
                                __m256 xv3 = _mm256_loadu_ps(x + col + 3 * lanes);
                                __m256 av0 = _mm256_loadu_ps(a + col + 0 * lanes);
                                __m256 av1 = _mm256_loadu_ps(a + col + 1 * lanes);
                                __m256 av2 = _mm256_loadu_ps(a + col + 2 * lanes);
                                __m256 av3 = _mm256_loadu_ps(a + col + 3 * lanes);
                                acc0 = _mm256_fmadd_ps(av0, xv0, acc0);
                                acc1 = _mm256_fmadd_ps(av1, xv1, acc1);
                                acc2 = _mm256_fmadd_ps(av2, xv2, acc2);
                                acc3 = _mm256_fmadd_ps(av3, xv3, acc3);
                            }
                            __m256 acc01 = _mm256_add_ps(acc0, acc1);
                            __m256 acc23 = _mm256_add_ps(acc2, acc3);
                            __m256 acc = _mm256_add_ps(acc01, acc23);
                            for (; col < upperVec; col += lanes) {
                                __m256 xv = _mm256_loadu_ps(x + col);
                                __m256 av = _mm256_loadu_ps(a + col);
                                acc = _mm256_fmadd_ps(av, xv, acc);
                            }
                            float dot = hsum256_ps(acc);
                            for (; col < N; col++) {
                                dot += a[col] * x[col];
                            }
                            y[row] = dot;
                        }
                    }
                    #endif

                    void %s(void **buffers, uint64_t *scalars, uint64_t scratch) {
                        (void)scratch;
                        const float *A = (const float *)buffers[0];
                        const float *x = (const float *)buffers[1];
                        float *y = (float *)buffers[2];
                        int M = (int)scalars[0];
                        int N = (int)scalars[1];

                        if (M <= 0 || N <= 0 || A == NULL || x == NULL || y == NULL) {
                            return;
                        }

                    #if defined(__x86_64__) || defined(__i386__)
                      #if defined(__GNUC__) || defined(__clang__)
                        if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
                            gemv_avx2_fma(A, x, y, M, N);
                            return;
                        }
                      #endif
                    #endif
                        gemv_scalar(A, x, y, M, N);
                    }
                    """,
                    minRows,
                    chunk,
                    prefetch,
                    functionName);
        }

        private static String buildCGemvPairSwigluKernelSource(
                String functionName, int ompMinRows, int ompChunk, int prefetchFloats) {
            int minRows = Math.max(1, ompMinRows);
            int chunk = Math.max(1, ompChunk);
            int prefetch = Math.max(16, prefetchFloats);
            return String.format(
                    Locale.ROOT,
                    """
                    #include <stdint.h>
                    #include <stddef.h>
                    #include <math.h>
                    #if defined(__x86_64__) || defined(__i386__)
                    #include <immintrin.h>
                    #endif

                    #define LLAMA_OMP_MIN_ROWS %d
                    #define LLAMA_OMP_CHUNK %d
                    #define LLAMA_PREFETCH_FLOATS %d

                    static inline float swiglu_scalar(float g, float u) {
                        return (g / (1.0f + expf(-g))) * u;
                    }

                    static inline void gemv_pair_scalar(
                            const float *A0,
                            const float *A1,
                            const float *x,
                            float *y,
                            int M,
                            int N) {
                    #if defined(_OPENMP)
                        #pragma omp parallel for schedule(static, LLAMA_OMP_CHUNK) if(M >= LLAMA_OMP_MIN_ROWS)
                    #endif
                        for (int row = 0; row < M; row++) {
                            const float *a0 = A0 + (size_t) row * (size_t) N;
                            const float *a1 = A1 + (size_t) row * (size_t) N;
                            float d0 = 0.0f;
                            float d1 = 0.0f;
                            for (int col = 0; col < N; col++) {
                                float xv = x[col];
                                d0 += a0[col] * xv;
                                d1 += a1[col] * xv;
                            }
                            y[row] = swiglu_scalar(d0, d1);
                        }
                    }

                    #if defined(__x86_64__) || defined(__i386__)
                    static inline float hsum256_ps(__m256 v) {
                        __m128 lo = _mm256_castps256_ps128(v);
                        __m128 hi = _mm256_extractf128_ps(v, 1);
                        __m128 sum = _mm_add_ps(lo, hi);
                        __m128 shuf = _mm_movehdup_ps(sum);
                        __m128 sums = _mm_add_ps(sum, shuf);
                        shuf = _mm_movehl_ps(shuf, sums);
                        sums = _mm_add_ss(sums, shuf);
                        return _mm_cvtss_f32(sums);
                    }

                    #if defined(__GNUC__) || defined(__clang__)
                    __attribute__((target("avx2,fma")))
                    #endif
                    static void gemv_pair_avx2_fma(
                            const float *A0,
                            const float *A1,
                            const float *x,
                            float *y,
                            int M,
                            int N) {
                        const int lanes = 8;
                        const int unroll = 4;
                        const int step = lanes * unroll;
                        int upper = N - (N %% step);
                        int upperVec = N - (N %% lanes);
                    #if defined(_OPENMP)
                        #pragma omp parallel for schedule(static, LLAMA_OMP_CHUNK) if(M >= LLAMA_OMP_MIN_ROWS)
                    #endif
                        for (int row = 0; row < M; row++) {
                            const float *a0 = A0 + (size_t) row * (size_t) N;
                            const float *a1 = A1 + (size_t) row * (size_t) N;
                            __m256 a00 = _mm256_setzero_ps(), a01 = _mm256_setzero_ps();
                            __m256 a02 = _mm256_setzero_ps(), a03 = _mm256_setzero_ps();
                            __m256 b00 = _mm256_setzero_ps(), b01 = _mm256_setzero_ps();
                            __m256 b02 = _mm256_setzero_ps(), b03 = _mm256_setzero_ps();
                            int col = 0;
                            for (; col < upper; col += step) {
                    #if defined(__GNUC__) || defined(__clang__)
                                __builtin_prefetch(a0 + col + LLAMA_PREFETCH_FLOATS, 0, 1);
                                __builtin_prefetch(a1 + col + LLAMA_PREFETCH_FLOATS, 0, 1);
                                __builtin_prefetch(x + col + LLAMA_PREFETCH_FLOATS, 0, 1);
                    #endif
                                __m256 xv0 = _mm256_loadu_ps(x + col + 0 * lanes);
                                __m256 xv1 = _mm256_loadu_ps(x + col + 1 * lanes);
                                __m256 xv2 = _mm256_loadu_ps(x + col + 2 * lanes);
                                __m256 xv3 = _mm256_loadu_ps(x + col + 3 * lanes);

                                __m256 av00 = _mm256_loadu_ps(a0 + col + 0 * lanes);
                                __m256 av01 = _mm256_loadu_ps(a0 + col + 1 * lanes);
                                __m256 av02 = _mm256_loadu_ps(a0 + col + 2 * lanes);
                                __m256 av03 = _mm256_loadu_ps(a0 + col + 3 * lanes);
                                a00 = _mm256_fmadd_ps(av00, xv0, a00);
                                a01 = _mm256_fmadd_ps(av01, xv1, a01);
                                a02 = _mm256_fmadd_ps(av02, xv2, a02);
                                a03 = _mm256_fmadd_ps(av03, xv3, a03);

                                __m256 bv00 = _mm256_loadu_ps(a1 + col + 0 * lanes);
                                __m256 bv01 = _mm256_loadu_ps(a1 + col + 1 * lanes);
                                __m256 bv02 = _mm256_loadu_ps(a1 + col + 2 * lanes);
                                __m256 bv03 = _mm256_loadu_ps(a1 + col + 3 * lanes);
                                b00 = _mm256_fmadd_ps(bv00, xv0, b00);
                                b01 = _mm256_fmadd_ps(bv01, xv1, b01);
                                b02 = _mm256_fmadd_ps(bv02, xv2, b02);
                                b03 = _mm256_fmadd_ps(bv03, xv3, b03);
                            }
                            __m256 da = _mm256_add_ps(_mm256_add_ps(a00, a01), _mm256_add_ps(a02, a03));
                            __m256 db = _mm256_add_ps(_mm256_add_ps(b00, b01), _mm256_add_ps(b02, b03));
                            for (; col < upperVec; col += lanes) {
                                __m256 xv = _mm256_loadu_ps(x + col);
                                __m256 av = _mm256_loadu_ps(a0 + col);
                                __m256 bv = _mm256_loadu_ps(a1 + col);
                                da = _mm256_fmadd_ps(av, xv, da);
                                db = _mm256_fmadd_ps(bv, xv, db);
                            }
                            float d0 = hsum256_ps(da);
                            float d1 = hsum256_ps(db);
                            for (; col < N; col++) {
                                float xv = x[col];
                                d0 += a0[col] * xv;
                                d1 += a1[col] * xv;
                            }
                            y[row] = swiglu_scalar(d0, d1);
                        }
                    }
                    #endif

                    void %s(void **buffers, uint64_t *scalars, uint64_t scratch) {
                        (void)scratch;
                        const float *A0 = (const float *)buffers[0];
                        const float *A1 = (const float *)buffers[1];
                        const float *x = (const float *)buffers[2];
                        float *y = (float *)buffers[3];
                        int M = (int)scalars[0];
                        int N = (int)scalars[1];

                        if (M <= 0 || N <= 0 || A0 == NULL || A1 == NULL || x == NULL || y == NULL) {
                            return;
                        }

                    #if defined(__x86_64__) || defined(__i386__)
                      #if defined(__GNUC__) || defined(__clang__)
                        if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
                            gemv_pair_avx2_fma(A0, A1, x, y, M, N);
                            return;
                        }
                      #endif
                    #endif
                        gemv_pair_scalar(A0, A1, x, y, M, N);
                    }
                    """,
                    minRows,
                    chunk,
                    prefetch,
                    functionName);
        }

        private static final boolean TRACE = Boolean.getBoolean("com.qxotic.trace");
        private static final int TRACE_TOKEN_LIMIT =
                Integer.getInteger("com.qxotic.trace.tokenLimit", 8);
        private static final boolean USE_C_GEMV =
                Boolean.parseBoolean(System.getProperty("com.qxotic.llama.cgemv", "false"));
        private static final long C_GEMV_MIN_WORK_GENERAL =
                Long.getLong("com.qxotic.llama.cgemv.minWork.general", Long.MAX_VALUE);
        private static final long C_GEMV_MIN_WORK_DOWN =
                Long.getLong("com.qxotic.llama.cgemv.minWork.down", 262_144L);
        private static final long C_GEMV_MIN_WORK_LOGITS =
                Long.getLong("com.qxotic.llama.cgemv.minWork.logits", 131_072L);
        private static final long C_GEMV_MIN_WORK_PAIR =
                Long.getLong("com.qxotic.llama.cgemv.minWork.pair", 262_144L);

        private enum GemvRoute {
            GENERAL,
            DOWN,
            LOGITS
        }

        private final LlamaConfig cfg;
        private final LlamaWeights w;
        private final DeviceRuntime runtime;
        private final DeviceRuntime cRuntime;
        private final KernelExecutable gemvKernel;
        private final KernelExecutable gemv2Kernel;
        private final KernelExecutable gemv3Kernel;
        private final KernelExecutable gemvDownKernel;
        private final KernelExecutable swigluKernel;
        private final KernelExecutable cGemvPairKernel;
        private final KernelExecutable cGemvDownKernel;
        private final KernelExecutable cGemvLogitsKernel;
        private final ExecutionStream stream;
        private final ExecutionStream cStream;
        private final MemoryView<?>[] keyCache;
        private final MemoryView<?>[] valueCache;
        private final MemoryDomain<?> domain;
        private final Tensor normReduceOnes;
        private final Tensor normBroadcastOnes;
        // [contextLen, 1] ones used to reduce [1, len] scores into [1, 1] via matmul.
        private final Tensor contextReduceOnes;
        // [1, contextLen] ones used to expand [1, 1] back to [1, len] via matmul.
        private final Tensor contextBroadcastOnes;
        private int position;
        private long gemvTimeNs;
        private long gemvPrepNs;
        private long gemvLaunchNs;
        private long gemvCalls;
        private long gemvFallbackCalls;
        private long gemvMissDtype;
        private long gemvMissXShape;
        private long gemvMissWShape;
        private long gemvMissShapeMismatch;
        private long gemvMissDevice;
        private long gemvMissContiguous;
        private long gemvMissKernelUnavailable;
        private long forwardTimeNs;
        private long forwardCalls;
        private long attentionTimeNs;
        private long attentionNormNs;
        private long ropeTimeNs;
        private long kvCopyTimeNs;
        private long qkMatmulNs;
        private long softmaxNs;
        private long pvMatmulNs;
        private long attentionScatterNs;
        private long attentionProjNs;
        private long ffnTimeNs;
        private long ffnNormNs;
        private long qkvProjNs;
        private long ffnPairProjNs;
        private long ffnSwigluNs;
        private long ffnGateProjNs;
        private long ffnUpProjNs;
        private long ffnActivationNs;
        private long ffnMulNs;
        private long ffnDownProjNs;
        private long ffnDownSpecialNs;
        private long logitsNs;

        Llama32Model(LlamaConfig cfg, LlamaWeights w) {
            this.cfg = cfg;
            this.w = w;
            this.runtime = Environment.runtimeFor(Device.defaultDevice());
            DeviceRuntime cRt;
            try {
                cRt = Environment.runtimeFor(DeviceType.C.deviceIndex(0));
            } catch (RuntimeException ex) {
                cRt = null;
            }
            this.cRuntime = cRt;
            this.gemvKernel = registerGemvKernel(runtime);
            this.gemv2Kernel = registerGemv2Kernel(runtime);
            this.gemv3Kernel = registerGemv3Kernel(runtime);
            this.gemvDownKernel = registerGemvDownKernel(runtime);
            this.swigluKernel = registerSwigluKernel(runtime);
            this.cGemvPairKernel =
                    registerCGemvKernel(
                            cRuntime,
                            C_GEMV_PAIR_KERNEL_NAME,
                            C_GEMV_PAIR_KERNEL_SOURCE,
                            "llama_gemv_pair_swiglu_avx2");
            this.cGemvDownKernel =
                    registerCGemvKernel(
                            cRuntime,
                            C_GEMV_DOWN_KERNEL_NAME,
                            C_GEMV_DOWN_KERNEL_SOURCE,
                            "llama_gemv_down_avx2");
            this.cGemvLogitsKernel =
                    registerCGemvKernel(
                            cRuntime,
                            C_GEMV_LOGITS_KERNEL_NAME,
                            C_GEMV_LOGITS_KERNEL_SOURCE,
                            "llama_gemv_logits_avx2");
            this.stream = new ExecutionStream(runtime.device(), null, true);
            this.cStream =
                    cRuntime != null ? new ExecutionStream(cRuntime.device(), null, true) : null;
            int kvDim = cfg.nKvHeads * cfg.headDim;
            this.keyCache = new MemoryView<?>[cfg.nLayers];
            this.valueCache = new MemoryView<?>[cfg.nLayers];
            this.domain = runtime.memoryDomain();
            System.err.println(
                    "[KV Cache] contextLength="
                            + cfg.contextLength
                            + " kvDim="
                            + kvDim
                            + " (nKvHeads="
                            + cfg.nKvHeads
                            + " × headDim="
                            + cfg.headDim
                            + ")");
            for (int i = 0; i < cfg.nLayers; i++) {
                this.keyCache[i] =
                        MemoryView.of(
                                this.domain
                                        .memoryAllocator()
                                        .allocateMemory(
                                                DataType.FP32, Shape.of(cfg.contextLength, kvDim)),
                                DataType.FP32,
                                Layout.rowMajor(Shape.of(cfg.contextLength, kvDim)));
                Memory<?> valueMemory =
                        this.domain
                                .memoryAllocator()
                                .allocateMemory(DataType.FP32, Shape.of(kvDim, cfg.contextLength));
                this.valueCache[i] =
                        MemoryView.of(
                                valueMemory,
                                DataType.FP32,
                                Layout.rowMajor(Shape.of(kvDim, cfg.contextLength)));
                if (i == 0) {
                    System.err.println(
                            "[KV Cache] keyCache[0] shape="
                                    + this.keyCache[0].shape()
                                    + " stride="
                                    + java.util.Arrays.toString(
                                            this.keyCache[0].layout().stride().toArray()));
                    System.err.println(
                            "[KV Cache] valueCache[0] shape="
                                    + this.valueCache[0].shape()
                                    + " stride="
                                    + java.util.Arrays.toString(
                                            this.valueCache[0].layout().stride().toArray()));
                    System.err.println(
                            "[KV Cache] valueCache[0] allocated bytes="
                                    + (kvDim * cfg.contextLength * 4L)
                                    + " base="
                                    + valueMemory.base());
                }
            }
            float[] ones = new float[cfg.dim];
            Arrays.fill(ones, 1f);
            this.normReduceOnes = Tensor.of(ones, Shape.of(cfg.dim, 1));
            this.normBroadcastOnes = Tensor.of(ones, Shape.of(1, cfg.dim));
            float[] contextOnes = new float[cfg.contextLength];
            Arrays.fill(contextOnes, 1f);
            this.contextReduceOnes = Tensor.of(contextOnes, Shape.of(cfg.contextLength, 1));
            this.contextBroadcastOnes = Tensor.of(contextOnes, Shape.of(1, cfg.contextLength));
            preTouchModelMemory();
            this.position = 0;
        }

        DecodeScratch createDecodeScratch() {
            DecodeScratch scratch = new DecodeScratch(cfg, domain);
            preTouchDecodeScratch(scratch);
            return scratch;
        }

        private static final class DecodeScratch {
            final MemoryView<?> xBuf;
            final MemoryView<?> xTmpBuf;
            final MemoryView<?> xbBuf;
            final MemoryView<?> ffInBuf;
            final MemoryView<?> qPreBuf;
            final MemoryView<?> kPreBuf;
            final MemoryView<?> vPreBuf;
            final MemoryView<?> qRotBuf;
            final MemoryView<?> kRotBuf;
            final MemoryView<?> attnScoresBuf;
            final MemoryView<?> attentionOutBuf;
            final MemoryView<?> ffnGateBuf;
            final MemoryView<?> ffnUpBuf;
            final MemoryView<?> ffnHiddenBuf;
            final MemoryView<?> ffnDownBuf;
            final MemoryView<?> logitsBuf;
            final Tensor xTensor;
            final Tensor xTmpTensor;
            final Tensor xbTensor;
            final Tensor ffInTensor;
            final Tensor qPreTensor;
            final Tensor kPreTensor;
            final Tensor vPreTensor;
            final Tensor qRotTensor;
            final Tensor kRotTensor;
            final Tensor attentionOutTensor;
            final Tensor ffnGateTensor;
            final Tensor ffnUpTensor;
            final Tensor ffnHiddenTensor;
            final Tensor ffnDownTensor;
            final Tensor logitsTensor;

            DecodeScratch(LlamaConfig cfg, MemoryDomain<?> domain) {
                int kvDim = cfg.nKvHeads * cfg.headDim;
                long xBytes = (long) cfg.dim * Float.BYTES;
                long xTmpBytes = (long) cfg.dim * Float.BYTES;
                long xbBytes = (long) cfg.dim * Float.BYTES;
                long ffInBytes = (long) cfg.dim * Float.BYTES;
                long qPreBytes = (long) cfg.dim * Float.BYTES;
                long kPreBytes = (long) kvDim * Float.BYTES;
                long vPreBytes = (long) kvDim * Float.BYTES;
                long qRotBytes = (long) cfg.dim * Float.BYTES;
                long kRotBytes = (long) kvDim * Float.BYTES;
                long attnScoresBytes = (long) cfg.contextLength * Float.BYTES;
                long attnOutBytes = (long) cfg.nHeads * cfg.headDim * Float.BYTES;
                long ffnGateBytes = (long) cfg.ffnDim * Float.BYTES;
                long ffnUpBytes = (long) cfg.ffnDim * Float.BYTES;
                long ffnHiddenBytes = (long) cfg.ffnDim * Float.BYTES;
                long ffnDownBytes = (long) cfg.dim * Float.BYTES;
                long logitsBytes = (long) cfg.vocabularySize * Float.BYTES;

                long offBytes = 0L;
                long offX = alignUp(offBytes, PAGE_BYTES);
                offBytes = offX + xBytes;
                long offXTmp = alignUp(offBytes, PAGE_BYTES);
                offBytes = offXTmp + xTmpBytes;
                long offXb = alignUp(offBytes, PAGE_BYTES);
                offBytes = offXb + xbBytes;
                long offFfIn = alignUp(offBytes, PAGE_BYTES);
                offBytes = offFfIn + ffInBytes;
                long offQPre = alignUp(offBytes, PAGE_BYTES);
                offBytes = offQPre + qPreBytes;
                long offKPre = alignUp(offBytes, PAGE_BYTES);
                offBytes = offKPre + kPreBytes;
                long offVPre = alignUp(offBytes, PAGE_BYTES);
                offBytes = offVPre + vPreBytes;
                long offQRot = alignUp(offBytes, PAGE_BYTES);
                offBytes = offQRot + qRotBytes;
                long offKRot = alignUp(offBytes, PAGE_BYTES);
                offBytes = offKRot + kRotBytes;
                long offAttnScores = alignUp(offBytes, PAGE_BYTES);
                offBytes = offAttnScores + attnScoresBytes;
                long offAttnOut = alignUp(offBytes, PAGE_BYTES);
                offBytes = offAttnOut + attnOutBytes;
                long offFfnGate = alignUp(offBytes, PAGE_BYTES);
                offBytes = offFfnGate + ffnGateBytes;
                long offFfnUp = alignUp(offBytes, PAGE_BYTES);
                offBytes = offFfnUp + ffnUpBytes;
                long offFfnHidden = alignUp(offBytes, PAGE_BYTES);
                offBytes = offFfnHidden + ffnHiddenBytes;
                long offFfnDown = alignUp(offBytes, PAGE_BYTES);
                offBytes = offFfnDown + ffnDownBytes;
                long offLogits = alignUp(offBytes, PAGE_BYTES);
                offBytes = offLogits + logitsBytes;

                Memory<?> scratchMemory =
                        domain.memoryAllocator().allocateMemory(offBytes, PAGE_BYTES);
                this.xBuf =
                        MemoryView.of(
                                scratchMemory,
                                offX,
                                DataType.FP32,
                                Layout.rowMajor(Shape.of(1, cfg.dim)));
                this.xTmpBuf =
                        MemoryView.of(
                                scratchMemory,
                                offXTmp,
                                DataType.FP32,
                                Layout.rowMajor(Shape.of(1, cfg.dim)));
                this.xbBuf =
                        MemoryView.of(
                                scratchMemory,
                                offXb,
                                DataType.FP32,
                                Layout.rowMajor(Shape.of(1, cfg.dim)));
                this.ffInBuf =
                        MemoryView.of(
                                scratchMemory,
                                offFfIn,
                                DataType.FP32,
                                Layout.rowMajor(Shape.of(1, cfg.dim)));
                this.qPreBuf =
                        MemoryView.of(
                                scratchMemory,
                                offQPre,
                                DataType.FP32,
                                Layout.rowMajor(Shape.of(1, cfg.dim)));
                this.kPreBuf =
                        MemoryView.of(
                                scratchMemory,
                                offKPre,
                                DataType.FP32,
                                Layout.rowMajor(Shape.of(1, kvDim)));
                this.vPreBuf =
                        MemoryView.of(
                                scratchMemory,
                                offVPre,
                                DataType.FP32,
                                Layout.rowMajor(Shape.of(1, kvDim)));
                this.qRotBuf =
                        MemoryView.of(
                                scratchMemory,
                                offQRot,
                                DataType.FP32,
                                Layout.rowMajor(Shape.of(1, cfg.dim)));
                this.kRotBuf =
                        MemoryView.of(
                                scratchMemory,
                                offKRot,
                                DataType.FP32,
                                Layout.rowMajor(Shape.of(1, kvDim)));
                this.attnScoresBuf =
                        MemoryView.of(
                                scratchMemory,
                                offAttnScores,
                                DataType.FP32,
                                Layout.rowMajor(Shape.of(cfg.contextLength)));
                this.attentionOutBuf =
                        MemoryView.of(
                                scratchMemory,
                                offAttnOut,
                                DataType.FP32,
                                Layout.rowMajor(Shape.of(1, cfg.nHeads * cfg.headDim)));
                this.ffnGateBuf =
                        MemoryView.of(
                                scratchMemory,
                                offFfnGate,
                                DataType.FP32,
                                Layout.rowMajor(Shape.of(1, cfg.ffnDim)));
                this.ffnUpBuf =
                        MemoryView.of(
                                scratchMemory,
                                offFfnUp,
                                DataType.FP32,
                                Layout.rowMajor(Shape.of(1, cfg.ffnDim)));
                this.ffnHiddenBuf =
                        MemoryView.of(
                                scratchMemory,
                                offFfnHidden,
                                DataType.FP32,
                                Layout.rowMajor(Shape.of(1, cfg.ffnDim)));
                this.ffnDownBuf =
                        MemoryView.of(
                                scratchMemory,
                                offFfnDown,
                                DataType.FP32,
                                Layout.rowMajor(Shape.of(1, cfg.dim)));
                this.logitsBuf =
                        MemoryView.of(
                                scratchMemory,
                                offLogits,
                                DataType.FP32,
                                Layout.rowMajor(Shape.of(cfg.vocabularySize)));

                this.xTensor = Tensor.of(xBuf);
                this.xTmpTensor = Tensor.of(xTmpBuf);
                this.xbTensor = Tensor.of(xbBuf);
                this.ffInTensor = Tensor.of(ffInBuf);
                this.qPreTensor = Tensor.of(qPreBuf);
                this.kPreTensor = Tensor.of(kPreBuf);
                this.vPreTensor = Tensor.of(vPreBuf);
                this.qRotTensor = Tensor.of(qRotBuf);
                this.kRotTensor = Tensor.of(kRotBuf);
                this.attentionOutTensor = Tensor.of(attentionOutBuf);
                this.ffnGateTensor = Tensor.of(ffnGateBuf);
                this.ffnUpTensor = Tensor.of(ffnUpBuf);
                this.ffnHiddenTensor = Tensor.of(ffnHiddenBuf);
                this.ffnDownTensor = Tensor.of(ffnDownBuf);
                this.logitsTensor = Tensor.of(logitsBuf);
            }

            private static long alignUp(long value, long alignment) {
                long mask = alignment - 1;
                return (value + mask) & ~mask;
            }
        }

        MemoryView<?> forward(int token, DecodeScratch scratch) {
            System.err.println("[forward] START position=" + position + " token=" + token);
            long forwardStartNs = System.nanoTime();
            System.err.println("[forward] loading embedding...");
            loadEmbeddingIntoScratch(token, scratch);
            System.err.println("[forward] embedding loaded");
            Tensor xTensor = scratch.xTensor;
            Tensor xTmpTensor = scratch.xTmpTensor;
            traceTensor("x_embed", -1, position, xTensor);
            int kvDim = cfg.nKvHeads * cfg.headDim;

            System.err.println("[forward] starting layer loop");
            for (int layer = 0; layer < cfg.nLayers; layer++) {
                long attentionStartNs = System.nanoTime();
                long t0 = System.nanoTime();
                rmsNormInto(xTensor, w.attnNorm[layer], scratch.xbBuf);
                attentionNormNs += (System.nanoTime() - t0);
                traceTensor("xb", layer, position, scratch.xbTensor);
                t0 = System.nanoTime();
                projectQkvInto(scratch.xbTensor, w.wq[layer], w.wk[layer], w.wv[layer], scratch);
                qkvProjNs += (System.nanoTime() - t0);
                Tensor qPreTensor = scratch.qPreTensor;
                Tensor kPreTensor = scratch.kPreTensor;
                Tensor vPreTensor = scratch.vPreTensor;
                traceTensor("q_pre", layer, position, qPreTensor);
                traceTensor("k_pre", layer, position, kPreTensor);
                traceTensor("v_pre", layer, position, vPreTensor);
                t0 = System.nanoTime();
                Tensor q =
                        applyRoPETensor(
                                qPreTensor,
                                cfg.nHeads,
                                position,
                                w.rope,
                                scratch.qRotBuf,
                                scratch.qRotTensor);
                Tensor k =
                        applyRoPETensor(
                                kPreTensor,
                                cfg.nKvHeads,
                                position,
                                w.rope,
                                scratch.kRotBuf,
                                scratch.kRotTensor);
                ropeTimeNs += (System.nanoTime() - t0);
                Tensor v = vPreTensor;

                traceTensor("q", layer, position, q);
                traceTensor("k", layer, position, k);

                System.err.println(
                        "[forward] About to write KV cache: layer="
                                + layer
                                + " position="
                                + position);
                t0 = System.nanoTime();
                writeKeyValueCache(layer, position, kvDim, k, v);
                kvCopyTimeNs += (System.nanoTime() - t0);

                Tensor attn = attention(q, layer, position + 1, scratch);
                traceTensor("attn_out", layer, position, attn);
                t0 = System.nanoTime();
                projectInto(attn, w.wo[layer], scratch.ffnDownBuf, GemvRoute.GENERAL);
                addInto(scratch.xBuf, scratch.ffnDownBuf, scratch.xTmpBuf, cfg.dim);
                attentionProjNs += (System.nanoTime() - t0);
                traceTensor("x", layer, position, xTmpTensor);
                attentionTimeNs += (System.nanoTime() - attentionStartNs);

                long ffnStartNs = System.nanoTime();
                t0 = System.nanoTime();
                rmsNormInto(xTmpTensor, w.ffnNorm[layer], scratch.ffInBuf);
                ffnNormNs += (System.nanoTime() - t0);
                t0 = System.nanoTime();
                projectPairSwigluInto(scratch.ffInTensor, w.wGate[layer], w.wUp[layer], scratch);
                ffnPairProjNs += (System.nanoTime() - t0);
                t0 = System.nanoTime();
                projectDownInto(scratch.ffnHiddenTensor, w.wDown[layer], scratch.ffnDownBuf);
                addInto(scratch.xTmpBuf, scratch.ffnDownBuf, scratch.xBuf, cfg.dim);
                ffnDownProjNs += (System.nanoTime() - t0);
                ffnTimeNs += (System.nanoTime() - ffnStartNs);
            }

            long logitsStartNs = System.nanoTime();
            rmsNormInto(xTensor, w.outputNorm, scratch.ffInBuf);
            projectInto(scratch.ffInTensor, w.output, scratch.logitsBuf, GemvRoute.LOGITS);
            logitsNs += (System.nanoTime() - logitsStartNs);
            traceTensor("logits", cfg.nLayers, position, scratch.logitsTensor);
            position++;
            long forwardEndNs = System.nanoTime();
            forwardTimeNs += (forwardEndNs - forwardStartNs);
            forwardCalls++;
            return scratch.logitsBuf;
        }

        void resetTimingStats() {
            gemvTimeNs = 0L;
            gemvPrepNs = 0L;
            gemvLaunchNs = 0L;
            gemvCalls = 0L;
            gemvFallbackCalls = 0L;
            gemvMissDtype = 0L;
            gemvMissXShape = 0L;
            gemvMissWShape = 0L;
            gemvMissShapeMismatch = 0L;
            gemvMissDevice = 0L;
            gemvMissContiguous = 0L;
            gemvMissKernelUnavailable = 0L;
            forwardTimeNs = 0L;
            forwardCalls = 0L;
            attentionTimeNs = 0L;
            attentionNormNs = 0L;
            ropeTimeNs = 0L;
            kvCopyTimeNs = 0L;
            qkMatmulNs = 0L;
            softmaxNs = 0L;
            pvMatmulNs = 0L;
            attentionScatterNs = 0L;
            attentionProjNs = 0L;
            ffnTimeNs = 0L;
            ffnNormNs = 0L;
            qkvProjNs = 0L;
            ffnPairProjNs = 0L;
            ffnSwigluNs = 0L;
            ffnGateProjNs = 0L;
            ffnUpProjNs = 0L;
            ffnActivationNs = 0L;
            ffnMulNs = 0L;
            ffnDownProjNs = 0L;
            ffnDownSpecialNs = 0L;
            logitsNs = 0L;
        }

        void resetDecodeState() {
            position = 0;
        }

        void printTimingSummary(String mode, long wallNs) {
            double wallMs = wallNs / 1_000_000.0;
            double forwardMs = forwardTimeNs / 1_000_000.0;
            double gemvMs = gemvTimeNs / 1_000_000.0;
            double attentionMs = attentionTimeNs / 1_000_000.0;
            double ffnMs = ffnTimeNs / 1_000_000.0;
            double logitsMs = logitsNs / 1_000_000.0;
            double gemvPctOfForward =
                    forwardTimeNs > 0 ? (100.0 * gemvTimeNs) / (double) forwardTimeNs : 0.0;
            double gemvPctOfWall = wallNs > 0 ? (100.0 * gemvTimeNs) / (double) wallNs : 0.0;
            double avgGemvUs = gemvCalls > 0 ? (gemvTimeNs / 1_000.0) / (double) gemvCalls : 0.0;
            System.out.printf(
                    "timing[%s]: wall=%.3fms forward=%.3fms gemv=%.3fms (%.2f%%) attention=%.3fms"
                            + " ffn=%.3fms logits=%.3fms calls=%d avgGemv=%.3fus gemv/wall=%.2f%%"
                            + " forwardCalls=%d%n",
                    mode,
                    wallMs,
                    forwardMs,
                    gemvMs,
                    gemvPctOfForward,
                    attentionMs,
                    ffnMs,
                    logitsMs,
                    gemvCalls,
                    avgGemvUs,
                    gemvPctOfWall,
                    forwardCalls);
            System.out.printf(
                    "timing_detail[%s]: gemvPrep=%.3fms gemvLaunch=%.3fms attnNorm=%.3fms"
                            + " qkvProj=%.3fms rope=%.3fms kvCopy=%.3fms qk=%.3fms softmax=%.3fms"
                            + " pv=%.3fms attnScatter=%.3fms attnProj=%.3fms ffnNorm=%.3fms"
                            + " ffnPairProj=%.3fms ffnSwiglu=%.3fms ffnGate=%.3fms ffnUp=%.3fms"
                            + " ffnAct=%.3fms ffnMul=%.3fms ffnDown=%.3fms ffnDownSpecial=%.3fms%n",
                    mode,
                    gemvPrepNs / 1_000_000.0,
                    gemvLaunchNs / 1_000_000.0,
                    attentionNormNs / 1_000_000.0,
                    qkvProjNs / 1_000_000.0,
                    ropeTimeNs / 1_000_000.0,
                    kvCopyTimeNs / 1_000_000.0,
                    qkMatmulNs / 1_000_000.0,
                    softmaxNs / 1_000_000.0,
                    pvMatmulNs / 1_000_000.0,
                    attentionScatterNs / 1_000_000.0,
                    attentionProjNs / 1_000_000.0,
                    ffnNormNs / 1_000_000.0,
                    ffnPairProjNs / 1_000_000.0,
                    ffnSwigluNs / 1_000_000.0,
                    ffnGateProjNs / 1_000_000.0,
                    ffnUpProjNs / 1_000_000.0,
                    ffnActivationNs / 1_000_000.0,
                    ffnMulNs / 1_000_000.0,
                    ffnDownProjNs / 1_000_000.0,
                    ffnDownSpecialNs / 1_000_000.0);
            System.out.printf(
                    "gemv_miss[%s]: fallback=%d dtype=%d xShape=%d wShape=%d shapeMismatch=%d"
                            + " device=%d contiguous=%d kernelUnavailable=%d%n",
                    mode,
                    gemvFallbackCalls,
                    gemvMissDtype,
                    gemvMissXShape,
                    gemvMissWShape,
                    gemvMissShapeMismatch,
                    gemvMissDevice,
                    gemvMissContiguous,
                    gemvMissKernelUnavailable);
        }

        private static void traceVector(String stage, int layer, int position, float[] data) {
            if (!TRACE || position >= TRACE_TOKEN_LIMIT) {
                return;
            }
            float sum = 0f;
            float sq = 0f;
            float maxAbs = 0f;
            for (float v : data) {
                sum += v;
                sq += v * v;
                maxAbs = Math.max(maxAbs, Math.abs(v));
            }
            float s0 = data.length > 0 ? data[0] : 0f;
            float s1 = data.length > 1 ? data[1] : 0f;
            System.out.printf(
                    "TRACE tensor vec stage=%s layer=%d pos=%d n=%d sum=%.6e l2=%.6e max=%.6e"
                            + " s0=%.6e s1=%.6e%n",
                    stage, layer, position, data.length, sum, Math.sqrt(sq), maxAbs, s0, s1);
        }

        private static void traceTensor(String stage, int layer, int position, Tensor tensor) {
            if (!TRACE || position >= TRACE_TOKEN_LIMIT) {
                return;
            }
            traceVector(stage, layer, position, toFloatArray(tensor));
        }

        private Tensor attention(Tensor q, int layer, int length, DecodeScratch scratch) {
            int kvMul = cfg.nHeads / cfg.nKvHeads;
            Tensor keySlice = Tensor.of(keyCache[layer].slice(0, 0, length));
            Tensor valueSlice = Tensor.of(valueCache[layer].slice(1, 0, length));
            MemoryView<?> outView = scratch.attentionOutBuf;

            // System.err.println("[attention] layer=" + layer + " length=" + length);

            float scale = (float) Math.sqrt(cfg.headDim);
            for (int kvHead = 0; kvHead < cfg.nKvHeads; kvHead++) {
                int h = kvHead * kvMul;
                Tensor qGroup =
                        q.slice(1, h * cfg.headDim, (h + kvMul) * cfg.headDim)
                                .view(Shape.of(kvMul, cfg.headDim));
                Tensor kGroup =
                        keySlice.slice(
                                1,
                                kvHead * cfg.headDim,
                                (kvHead + 1) * cfg.headDim); // [length, headDim]
                long t0 = System.nanoTime();
                Tensor scores =
                        qGroup.matmul(kGroup.transpose(0, 1)).divide(scale); // [kvMul, length]
                qkMatmulNs += (System.nanoTime() - t0);
                t0 = System.nanoTime();
                Tensor probs = softmaxRowsTensor(scores, kvMul, length);
                softmaxNs += (System.nanoTime() - t0);

                Tensor vGroupT =
                        valueSlice.slice(
                                0,
                                kvHead * cfg.headDim,
                                (kvHead + 1) * cfg.headDim); // [headDim, length]

                // WORKAROUND: Materialize vGroupT first to break the lazy computation graph
                // that has pathological strides from slice operations. Then create a fresh
                // Tensor from the materialized (row-major) view before transposing.
                // This prevents the LIR kernel from seeing strided inputs.
                MemoryView<?> vGroupTMaterialized = vGroupT.materialize();
                Tensor vGroupTFresh = Tensor.of(vGroupTMaterialized);

                t0 = System.nanoTime();
                Tensor outGroup = probs.matmul(vGroupTFresh.transpose(0, 1)); // [kvMul, headDim]
                MemoryView<?> outGroupView = outGroup.materialize();
                pvMatmulNs += (System.nanoTime() - t0);
                t0 = System.nanoTime();
                MemoryView<?> outViewSlice =
                        outView.slice(1, h * cfg.headDim, (h + kvMul) * cfg.headDim);
                copyTensorToView(outGroup.view(Shape.of(1, kvMul * cfg.headDim)), outViewSlice);
                attentionScatterNs += (System.nanoTime() - t0);
            }

            return scratch.attentionOutTensor;
        }

        private Tensor softmaxRowsTensor(Tensor x, int rows, int length) {
            Tensor exp = x.exp(); // [rows, len]
            Tensor denom = exp.matmul(contextReduceOnes.slice(0, 0, length)); // [rows, 1]
            Tensor denomExpanded =
                    denom.matmul(contextBroadcastOnes.slice(1, 0, length)); // [rows, len]
            return exp.divide(denomExpanded);
        }

        private void projectInto(Tensor x, Tensor weightOutIn, MemoryView<?> outBuffer) {
            projectInto(x, weightOutIn, outBuffer, GemvRoute.GENERAL);
        }

        private void projectInto(
                Tensor x, Tensor weightOutIn, MemoryView<?> outBuffer, GemvRoute route) {
            if (!projectWithGemvKernelInto(x, weightOutIn, outBuffer, route)) {
                gemvFallbackCalls++;
                projectWithDirectLoopInto(x, weightOutIn, outBuffer);
            }
        }

        private void projectQkvInto(
                Tensor x, Tensor wq, Tensor wk, Tensor wv, DecodeScratch scratch) {
            boolean fast =
                    projectQkvWithGemvKernelInto(
                            x, wq, wk, wv, scratch.qPreBuf, scratch.kPreBuf, scratch.vPreBuf);
            if (fast) {
                return;
            }
            projectInto(x, wq, scratch.qPreBuf);
            projectInto(x, wk, scratch.kPreBuf);
            projectInto(x, wv, scratch.vPreBuf);
        }

        private boolean projectQkvWithGemvKernelInto(
                Tensor x,
                Tensor wq,
                Tensor wk,
                Tensor wv,
                MemoryView<?> outQ,
                MemoryView<?> outK,
                MemoryView<?> outV) {
            if (gemv3Kernel == null
                    || x.dataType() != DataType.FP32
                    || wq.dataType() != DataType.FP32
                    || wk.dataType() != DataType.FP32
                    || wv.dataType() != DataType.FP32) {
                return false;
            }
            if (x.shape().rank() != 2 || x.shape().size(0) != 1 || x.shape().size(1) != cfg.dim) {
                return false;
            }
            if (wq.shape().rank() != 2
                    || wk.shape().rank() != 2
                    || wv.shape().rank() != 2
                    || wq.shape().size(1) != cfg.dim
                    || wk.shape().size(1) != cfg.dim
                    || wv.shape().size(1) != cfg.dim
                    || wq.shape().size(0) != cfg.dim
                    || wk.shape().size(0) != cfg.nKvHeads * cfg.headDim
                    || wv.shape().size(0) != cfg.nKvHeads * cfg.headDim) {
                return false;
            }

            long prepStartNs = System.nanoTime();
            MemoryView<?> q = wq.materialize();
            MemoryView<?> k = wk.materialize();
            MemoryView<?> v = wv.materialize();
            MemoryView<?> xv = x.materialize();
            if (!q.memory().device().equals(runtime.device())
                    || !k.memory().device().equals(runtime.device())
                    || !v.memory().device().equals(runtime.device())
                    || !xv.memory().device().equals(runtime.device())
                    || !outQ.memory().device().equals(runtime.device())
                    || !outK.memory().device().equals(runtime.device())
                    || !outV.memory().device().equals(runtime.device())) {
                return false;
            }
            if (!q.layout().isSuffixContiguous(0)
                    || !k.layout().isSuffixContiguous(0)
                    || !v.layout().isSuffixContiguous(0)
                    || !xv.layout().isSuffixContiguous(0)
                    || !outQ.layout().isSuffixContiguous(0)
                    || !outK.layout().isSuffixContiguous(0)
                    || !outV.layout().isSuffixContiguous(0)) {
                return false;
            }

            int mQ = cfg.dim;
            int mKV = cfg.nKvHeads * cfg.headDim;
            int nInt = cfg.dim;
            long prepEndNs = System.nanoTime();
            gemvPrepNs += (prepEndNs - prepStartNs);
            long gemvStartNs = prepEndNs;
            gemv3Kernel.launch(
                    LaunchConfig.auto(),
                    KernelArgs.fromVarargs(q, k, v, xv, outQ, outK, outV, mQ, mKV, nInt),
                    stream);
            long gemvEndNs = System.nanoTime();
            long elapsed = gemvEndNs - gemvStartNs;
            gemvTimeNs += elapsed;
            gemvLaunchNs += elapsed;
            gemvCalls += 3;
            return true;
        }

        private void projectDownInto(Tensor x, Tensor weightOutIn, MemoryView<?> outBuffer) {
            if (!projectDownWithGemvKernelInto(x, weightOutIn, outBuffer)) {
                projectInto(x, weightOutIn, outBuffer);
            }
        }

        private boolean projectDownWithGemvKernelInto(
                Tensor x, Tensor weightOutIn, MemoryView<?> outBuffer) {
            if ((gemvDownKernel == null && cGemvDownKernel == null)
                    || x.dataType() != DataType.FP32
                    || weightOutIn.dataType() != DataType.FP32) {
                return false;
            }
            if (x.shape().rank() != 2
                    || x.shape().size(0) != 1
                    || x.shape().size(1) != cfg.ffnDim) {
                return false;
            }
            if (weightOutIn.shape().rank() != 2
                    || weightOutIn.shape().size(0) != cfg.dim
                    || weightOutIn.shape().size(1) != cfg.ffnDim) {
                return false;
            }

            long prepStartNs = System.nanoTime();
            MemoryView<?> a = weightOutIn.materialize();
            MemoryView<?> xv = x.materialize();
            if (!a.memory().device().equals(runtime.device())
                    || !xv.memory().device().equals(runtime.device())
                    || !outBuffer.memory().device().equals(runtime.device())) {
                return false;
            }
            if (!a.layout().isSuffixContiguous(0)
                    || !xv.layout().isSuffixContiguous(0)
                    || !outBuffer.layout().isSuffixContiguous(0)) {
                return false;
            }
            int mInt = cfg.dim;
            int nInt = cfg.ffnDim;
            long prepEndNs = System.nanoTime();
            gemvPrepNs += (prepEndNs - prepStartNs);
            long gemvStartNs = prepEndNs;
            if (shouldUseCGemv(GemvRoute.DOWN, mInt, nInt)
                    && cGemvDownKernel != null
                    && cStream != null) {
                cGemvDownKernel.launch(
                        LaunchConfig.auto(),
                        KernelArgs.fromVarargs(a, xv, outBuffer, mInt, nInt),
                        cStream);
            } else {
                gemvDownKernel.launch(
                        LaunchConfig.auto(),
                        KernelArgs.fromVarargs(a, xv, outBuffer, mInt, nInt),
                        stream);
            }
            long gemvEndNs = System.nanoTime();
            long elapsed = gemvEndNs - gemvStartNs;
            gemvTimeNs += elapsed;
            gemvLaunchNs += elapsed;
            gemvCalls++;
            ffnDownSpecialNs += elapsed;
            return true;
        }

        private void projectPairSwigluInto(
                Tensor x, Tensor weightGate, Tensor weightUp, DecodeScratch scratch) {
            MemoryView<?> outHidden = scratch.ffnHiddenBuf;
            if (projectPairSwigluWithGemvKernelInto(x, weightGate, weightUp, outHidden)) {
                return;
            }
            projectInto(x, weightGate, scratch.ffnGateBuf);
            projectInto(x, weightUp, scratch.ffnUpBuf);
            if (applySwigluInto(scratch.ffnGateBuf, scratch.ffnUpBuf, outHidden)) {
                return;
            }
            applySwigluDirectInto(scratch.ffnGateBuf, scratch.ffnUpBuf, outHidden);
        }

        private boolean projectPairSwigluWithGemvKernelInto(
                Tensor x, Tensor weightGate, Tensor weightUp, MemoryView<?> outHidden) {
            if (x.dataType() != DataType.FP32
                    || weightGate.dataType() != DataType.FP32
                    || weightUp.dataType() != DataType.FP32) {
                return false;
            }
            if (x.shape().rank() != 2 || x.shape().size(0) != 1 || x.shape().size(1) != cfg.dim) {
                return false;
            }
            if (weightGate.shape().rank() != 2
                    || weightUp.shape().rank() != 2
                    || weightGate.shape().size(0) != cfg.ffnDim
                    || weightUp.shape().size(0) != cfg.ffnDim
                    || weightGate.shape().size(1) != cfg.dim
                    || weightUp.shape().size(1) != cfg.dim) {
                return false;
            }
            if (gemv2Kernel == null && cGemvPairKernel == null) {
                return false;
            }

            long prepStartNs = System.nanoTime();
            MemoryView<?> a0 = weightGate.materialize();
            MemoryView<?> a1 = weightUp.materialize();
            MemoryView<?> xv = x.materialize();
            if (!a0.memory().device().equals(runtime.device())
                    || !a1.memory().device().equals(runtime.device())
                    || !xv.memory().device().equals(runtime.device())
                    || !outHidden.memory().device().equals(runtime.device())) {
                return false;
            }
            if (!a0.layout().isSuffixContiguous(0)
                    || !a1.layout().isSuffixContiguous(0)
                    || !xv.layout().isSuffixContiguous(0)
                    || !outHidden.layout().isSuffixContiguous(0)) {
                return false;
            }
            int mInt = cfg.ffnDim;
            int nInt = cfg.dim;
            long prepEndNs = System.nanoTime();
            gemvPrepNs += (prepEndNs - prepStartNs);
            long gemvStartNs = prepEndNs;
            if (shouldUseCGemvPair(mInt, nInt) && cGemvPairKernel != null && cStream != null) {
                cGemvPairKernel.launch(
                        LaunchConfig.auto(),
                        KernelArgs.fromVarargs(a0, a1, xv, outHidden, mInt, nInt),
                        cStream);
            } else {
                gemv2Kernel.launch(
                        LaunchConfig.auto(),
                        KernelArgs.fromVarargs(a0, a1, xv, outHidden, mInt, nInt),
                        stream);
            }
            long gemvEndNs = System.nanoTime();
            long elapsed = gemvEndNs - gemvStartNs;
            gemvTimeNs += elapsed;
            gemvLaunchNs += elapsed;
            gemvCalls += 2;
            ffnSwigluNs += elapsed;
            return true;
        }

        private boolean applySwigluInto(
                MemoryView<?> gateView, MemoryView<?> upView, MemoryView<?> out) {
            if (swigluKernel == null
                    || gateView.dataType() != DataType.FP32
                    || upView.dataType() != DataType.FP32) {
                return false;
            }
            if (!gateView.memory().device().equals(runtime.device())
                    || !upView.memory().device().equals(runtime.device())
                    || !out.memory().device().equals(runtime.device())) {
                return false;
            }
            if (!gateView.layout().isSuffixContiguous(0)
                    || !upView.layout().isSuffixContiguous(0)
                    || !out.layout().isSuffixContiguous(0)) {
                return false;
            }
            long swigluStartNs = System.nanoTime();
            swigluKernel.launch(
                    LaunchConfig.auto(),
                    KernelArgs.fromVarargs(gateView, upView, out, cfg.ffnDim),
                    stream);
            long swigluEndNs = System.nanoTime();
            ffnSwigluNs += (swigluEndNs - swigluStartNs);
            return true;
        }

        private boolean projectWithGemvKernelInto(
                Tensor x, Tensor weightOutIn, MemoryView<?> outBuffer, GemvRoute route) {
            if (x.dataType() != DataType.FP32 || weightOutIn.dataType() != DataType.FP32) {
                gemvMissDtype++;
                return false;
            }
            if (x.shape().rank() != 2 || x.shape().size(0) != 1) {
                gemvMissXShape++;
                return false;
            }
            if (weightOutIn.shape().rank() != 2) {
                gemvMissWShape++;
                return false;
            }
            long m = weightOutIn.shape().size(0);
            long n = weightOutIn.shape().size(1);
            if (x.shape().size(1) != n) {
                gemvMissShapeMismatch++;
                return false;
            }

            long prepStartNs = System.nanoTime();
            MemoryView<?> a = weightOutIn.materialize();
            MemoryView<?> xv = x.materialize();
            if (!a.memory().device().equals(runtime.device())
                    || !xv.memory().device().equals(runtime.device())
                    || !outBuffer.memory().device().equals(runtime.device())) {
                gemvMissDevice++;
                return false;
            }
            if (!a.layout().isSuffixContiguous(0)
                    || !xv.layout().isSuffixContiguous(0)
                    || !outBuffer.layout().isSuffixContiguous(0)) {
                gemvMissContiguous++;
                return false;
            }
            KernelExecutable cKernel = cKernelForRoute(route);
            if (gemvKernel == null) {
                if (!shouldUseCGemv(route, m, n) || cKernel == null || cStream == null) {
                    gemvMissKernelUnavailable++;
                    return false;
                }
            }
            long prepEndNs = System.nanoTime();
            gemvPrepNs += (prepEndNs - prepStartNs);
            long gemvStartNs = prepEndNs;
            if (shouldUseCGemv(route, m, n) && cKernel != null && cStream != null) {
                cKernel.launch(
                        LaunchConfig.auto(),
                        KernelArgs.fromVarargs(
                                a, xv, outBuffer, Math.toIntExact(m), Math.toIntExact(n)),
                        cStream);
            } else {
                gemvKernel.launch(
                        LaunchConfig.auto(),
                        KernelArgs.fromVarargs(
                                a, xv, outBuffer, Math.toIntExact(m), Math.toIntExact(n)),
                        stream);
            }
            long gemvEndNs = System.nanoTime();
            gemvTimeNs += (gemvEndNs - gemvStartNs);
            gemvLaunchNs += (gemvEndNs - gemvStartNs);
            gemvCalls++;
            return true;
        }

        private KernelExecutable cKernelForRoute(GemvRoute route) {
            return switch (route) {
                case DOWN -> cGemvDownKernel;
                case LOGITS -> cGemvLogitsKernel;
                case GENERAL -> null;
            };
        }

        private static boolean shouldUseCGemv(GemvRoute route, long m, long n) {
            if (!USE_C_GEMV) {
                return false;
            }
            long work = m * n;
            return switch (route) {
                case GENERAL -> work >= C_GEMV_MIN_WORK_GENERAL;
                case DOWN -> work >= C_GEMV_MIN_WORK_DOWN;
                case LOGITS -> work >= C_GEMV_MIN_WORK_LOGITS;
            };
        }

        private static boolean shouldUseCGemvPair(long m, long n) {
            return USE_C_GEMV && (m * n) >= C_GEMV_MIN_WORK_PAIR;
        }

        private static KernelExecutable registerGemvKernel(DeviceRuntime runtime) {
            if (!runtime.supportsKernels()) {
                return null;
            }
            try {
                return runtime.loadRegisteredExecutable(GEMV_KERNEL_NAME)
                        .orElseGet(
                                () ->
                                        runtime.registerKernel(
                                                GEMV_KERNEL_NAME,
                                                KernelProgram.source(
                                                        "java",
                                                        GEMV_KERNEL_SOURCE,
                                                        "LlamaGemvKernel")));
            } catch (RuntimeException ex) {
                System.err.println(
                        "WARN: disabling fast llama.gemv.fp32 kernel; run JVM with --add-modules"
                                + " jdk.incubator.vector to enable Vector API. Cause: "
                                + ex.getMessage());
                return null;
            }
        }

        private static KernelExecutable registerGemv2Kernel(DeviceRuntime runtime) {
            if (!runtime.supportsKernels()) {
                return null;
            }
            try {
                return runtime.loadRegisteredExecutable(GEMV2_KERNEL_NAME)
                        .orElseGet(
                                () ->
                                        runtime.registerKernel(
                                                GEMV2_KERNEL_NAME,
                                                KernelProgram.source(
                                                        "java",
                                                        GEMV2_KERNEL_SOURCE,
                                                        "LlamaGemv2Kernel")));
            } catch (RuntimeException ex) {
                return null;
            }
        }

        private static KernelExecutable registerCGemvKernel(
                DeviceRuntime runtime, String kernelName, String kernelSource, String entryPoint) {
            if (runtime == null || !runtime.supportsKernels()) {
                return null;
            }
            try {
                return runtime.loadRegisteredExecutable(kernelName)
                        .orElseGet(
                                () ->
                                        runtime.registerKernel(
                                                kernelName,
                                                KernelProgram.source(
                                                        "c", kernelSource, entryPoint)));
            } catch (RuntimeException ex) {
                return null;
            }
        }

        private static KernelExecutable registerGemv3Kernel(DeviceRuntime runtime) {
            if (!runtime.supportsKernels()) {
                return null;
            }
            try {
                return runtime.loadRegisteredExecutable(GEMV3_KERNEL_NAME)
                        .orElseGet(
                                () ->
                                        runtime.registerKernel(
                                                GEMV3_KERNEL_NAME,
                                                KernelProgram.source(
                                                        "java",
                                                        GEMV3_KERNEL_SOURCE,
                                                        "LlamaGemv3Kernel")));
            } catch (RuntimeException ex) {
                return null;
            }
        }

        private static KernelExecutable registerGemvDownKernel(DeviceRuntime runtime) {
            if (!runtime.supportsKernels()) {
                return null;
            }
            try {
                return runtime.loadRegisteredExecutable(GEMV_DOWN_KERNEL_NAME)
                        .orElseGet(
                                () ->
                                        runtime.registerKernel(
                                                GEMV_DOWN_KERNEL_NAME,
                                                KernelProgram.source(
                                                        "java",
                                                        GEMV_DOWN_KERNEL_SOURCE,
                                                        "LlamaGemvDownKernel")));
            } catch (RuntimeException ex) {
                return null;
            }
        }

        private static KernelExecutable registerSwigluKernel(DeviceRuntime runtime) {
            if (!runtime.supportsKernels()) {
                return null;
            }
            try {
                return runtime.loadRegisteredExecutable(SWIGLU_KERNEL_NAME)
                        .orElseGet(
                                () ->
                                        runtime.registerKernel(
                                                SWIGLU_KERNEL_NAME,
                                                KernelProgram.source(
                                                        "java",
                                                        SWIGLU_KERNEL_SOURCE,
                                                        "LlamaSwigluKernel")));
            } catch (RuntimeException ex) {
                return null;
            }
        }

        @SuppressWarnings({"unchecked", "rawtypes"})
        private static void copyTensorToView(Tensor src, MemoryView<?> dst) {
            MemoryView<?> srcView = src.materialize();
            MemoryDomain srcDomain =
                    Environment.runtimeFor(srcView.memory().device()).memoryDomain();
            MemoryDomain dstDomain =
                    Environment.runtimeFor(dst.memory().device()).memoryDomain();
            MemoryDomain.copy(srcDomain, srcView, dstDomain, dst);
        }

        private void loadEmbeddingIntoScratch(int token, DecodeScratch scratch) {
            copyTensorToView(w.tokenTable.slice(0, token, token + 1), scratch.xBuf);
        }

        private void rmsNormInto(Tensor x, Tensor weight, MemoryView<?> out) {
            Tensor ms = x.multiply(x).matmul(normReduceOnes).divide((float) cfg.dim);
            Tensor inv = ms.add(cfg.rmsEps).sqrt().reciprocal().matmul(normBroadcastOnes);
            Tensor normalized = x.multiply(inv).multiply(weight.view(Shape.of(1, cfg.dim)));
            copyTensorToView(normalized, out);
        }

        private static void addInto(
                MemoryView<?> a, MemoryView<?> b, MemoryView<?> out, int length) {
            Tensor sum = Tensor.of(a).add(Tensor.of(b));
            copyTensorToView(sum, out);
        }

        private void projectWithDirectLoopInto(
                Tensor x, Tensor weightOutIn, MemoryView<?> outBuffer) {
            Tensor projected = weightOutIn.matmul(x.transpose(0, 1)).transpose(0, 1);
            copyTensorToView(projected.view(outBuffer.shape()), outBuffer);
        }

        private void applySwigluDirectInto(
                MemoryView<?> gate, MemoryView<?> up, MemoryView<?> out) {
            Tensor hidden = Tensor.of(gate).silu().multiply(Tensor.of(up));
            copyTensorToView(hidden, out);
        }

        private void writeKeyValueCache(
                int layer, int position, int kvDim, Tensor key, Tensor value) {
            copyTensorToView(key, keyCache[layer].slice(0, position, position + 1));

            // WORKAROUND: Use HipStridedCopy directly to copy with transpose
            // value is (1, 512), need to copy to valueCache[layer] at position with transpose
            // valueCache[layer] is (512, 2048), slice(1, position, position+1) gives (512, 1)
            MemoryView<?> valueMaterialized = value.materialize();
            MemoryView<?> dstSlice = valueCache[layer].slice(1, position, position + 1);

            // Manually transpose by swapping dimensions: (1, 512) -> (512, 1)
            MemoryView<?> valueTransposed = valueMaterialized.transpose(0, 1);

            MemoryDomain srcDomain =
                    Environment.runtimeFor(valueTransposed.memory().device())
                            .memoryDomain();
            MemoryDomain dstDomain =
                    Environment.runtimeFor(dstSlice.memory().device()).memoryDomain();
            MemoryDomain.copy(srcDomain, valueTransposed, dstDomain, dstSlice);
        }

        private Tensor applyRoPETensor(
                Tensor x,
                int heads,
                int pos,
                RopeTables rope,
                MemoryView<?> outBuffer,
                Tensor outTensor) {
            int half = cfg.headDim / 2;
            Tensor xHeads = x.view(Shape.of(heads, cfg.headDim));
            Tensor even = xHeads.matmul(rope.evenSelector);
            Tensor odd = xHeads.matmul(rope.oddSelector);

            Tensor cosRow = rope.cosTensor.slice(0, pos, pos + 1);
            Tensor sinRow = rope.sinTensor.slice(0, pos, pos + 1);
            Tensor headsOnes = rope.headsOnes.slice(0, 0, heads);
            Tensor cosBroadcast = headsOnes.matmul(cosRow);
            Tensor sinBroadcast = headsOnes.matmul(sinRow);

            Tensor re = even.multiply(cosBroadcast).subtract(odd.multiply(sinBroadcast));
            Tensor im = even.multiply(sinBroadcast).add(odd.multiply(cosBroadcast));

            Tensor outHeads = re.matmul(rope.evenScatter).add(im.matmul(rope.oddScatter));
            copyTensorToView(outHeads.view(Shape.of(1, heads * cfg.headDim)), outBuffer);
            return outTensor;
        }

        private Tensor applyRoPENeoxTensor(Tensor x, int heads, int pos, RopeTables rope) {
            float[] in = toFloatArray(x);
            float[] out = new float[in.length];
            float[] cos = rope.cos[pos];
            float[] sin = rope.sin[pos];
            int half = cfg.headDim / 2;
            for (int h = 0; h < heads; h++) {
                int base = h * cfg.headDim;
                for (int i = 0; i < half; i++) {
                    int reIdx = base + i;
                    int imIdx = base + half + i;
                    float re = in[reIdx];
                    float im = in[imIdx];
                    float c = cos[i];
                    float s = sin[i];
                    out[reIdx] = re * c - im * s;
                    out[imIdx] = re * s + im * c;
                }
            }
            return Tensor.of(out, Shape.of(1, heads * cfg.headDim));
        }

        private void preTouchModelMemory() {
            touchTensor(w.tokenTable);
            touchTensor(w.output);
            touchTensor(w.outputNorm);
            touchTensorArray(w.attnNorm);
            touchTensorArray(w.wq);
            touchTensorArray(w.wk);
            touchTensorArray(w.wv);
            touchTensorArray(w.wo);
            touchTensorArray(w.ffnNorm);
            touchTensorArray(w.wGate);
            touchTensorArray(w.wDown);
            touchTensorArray(w.wUp);

            for (MemoryView<?> cache : keyCache) {
                touchMemoryViewPages(cache);
            }
            for (MemoryView<?> cache : valueCache) {
                touchMemoryViewPages(cache);
            }
        }

        private void preTouchDecodeScratch(DecodeScratch scratch) {
            touchMemoryViewPages(scratch.xBuf);
            touchMemoryViewPages(scratch.xTmpBuf);
            touchMemoryViewPages(scratch.xbBuf);
            touchMemoryViewPages(scratch.ffInBuf);
            touchMemoryViewPages(scratch.qPreBuf);
            touchMemoryViewPages(scratch.kPreBuf);
            touchMemoryViewPages(scratch.vPreBuf);
            touchMemoryViewPages(scratch.qRotBuf);
            touchMemoryViewPages(scratch.kRotBuf);
            touchMemoryViewPages(scratch.attnScoresBuf);
            touchMemoryViewPages(scratch.attentionOutBuf);
            touchMemoryViewPages(scratch.ffnGateBuf);
            touchMemoryViewPages(scratch.ffnUpBuf);
            touchMemoryViewPages(scratch.ffnHiddenBuf);
            touchMemoryViewPages(scratch.ffnDownBuf);
            touchMemoryViewPages(scratch.logitsBuf);
        }

        private static void touchTensorArray(Tensor[] tensors) {
            for (Tensor tensor : tensors) {
                touchTensor(tensor);
            }
        }

        private static void touchTensor(Tensor tensor) {
            touchMemoryViewPages(tensor.materialize());
        }

        @SuppressWarnings({"rawtypes", "unchecked"})
        private static void touchMemoryViewPages(MemoryView<?> view) {
            if (view.dataType() != DataType.FP32) {
                return;
            }
            MemoryDomain domain =
                    Environment.runtimeFor(view.memory().device()).memoryDomain();
            MemoryAccess access = domain.directAccess();
            if (access == null) {
                return;
            }
            long elements = view.shape().size();
            if (elements <= 0) {
                return;
            }
            long step = Math.max(1L, PAGE_BYTES / Float.BYTES);
            float sum = 0f;
            for (long i = 0; i < elements; i += step) {
                long off = Indexing.linearToOffset(view, i);
                sum += access.readFloat(view.memory(), off);
            }
            long lastOff = Indexing.linearToOffset(view, elements - 1);
            sum += access.readFloat(view.memory(), lastOff);
            touchSink += sum;
        }
    }

    private static final class RopeTables {
        final float[][] cos;
        final float[][] sin;
        final Tensor cosTensor;
        final Tensor sinTensor;
        final Tensor headsOnes;
        final Tensor evenSelector;
        final Tensor oddSelector;
        final Tensor evenScatter;
        final Tensor oddScatter;
        final Tensor firstHalfScatter;
        final Tensor secondHalfScatter;

        private RopeTables(
                float[][] cos,
                float[][] sin,
                Tensor cosTensor,
                Tensor sinTensor,
                Tensor headsOnes,
                Tensor evenSelector,
                Tensor oddSelector,
                Tensor evenScatter,
                Tensor oddScatter,
                Tensor firstHalfScatter,
                Tensor secondHalfScatter) {
            this.cos = cos;
            this.sin = sin;
            this.cosTensor = cosTensor;
            this.sinTensor = sinTensor;
            this.headsOnes = headsOnes;
            this.evenSelector = evenSelector;
            this.oddSelector = oddSelector;
            this.evenScatter = evenScatter;
            this.oddScatter = oddScatter;
            this.firstHalfScatter = firstHalfScatter;
            this.secondHalfScatter = secondHalfScatter;
        }

        static RopeTables precompute(
                int ctx, int headDim, float theta, float[] ropeScales, int maxHeads) {
            int half = headDim / 2;
            float[][] cos = new float[ctx][half];
            float[][] sin = new float[ctx][half];
            float[] cosFlat = new float[ctx * half];
            float[] sinFlat = new float[ctx * half];
            int idx = 0;
            for (int p = 0; p < ctx; p++) {
                for (int i = 0; i < headDim; i += 2) {
                    int f = i / 2;
                    float freq = (float) (1.0 / Math.pow(theta, i / (double) headDim));
                    if (ropeScales != null) {
                        freq /= ropeScales[f];
                    }
                    float v = p * freq;
                    cos[p][f] = (float) Math.cos(v);
                    sin[p][f] = (float) Math.sin(v);
                    cosFlat[idx] = cos[p][f];
                    sinFlat[idx] = sin[p][f];
                    idx++;
                }
            }

            float[] headsOnes = new float[maxHeads];
            Arrays.fill(headsOnes, 1f);
            float[] evenSelector = new float[headDim * half];
            float[] oddSelector = new float[headDim * half];
            float[] evenScatter = new float[half * headDim];
            float[] oddScatter = new float[half * headDim];
            float[] firstHalfScatter = new float[half * headDim];
            float[] secondHalfScatter = new float[half * headDim];
            for (int j = 0; j < half; j++) {
                int evenCol = 2 * j;
                int oddCol = evenCol + 1;
                evenSelector[evenCol * half + j] = 1f;
                oddSelector[oddCol * half + j] = 1f;
                evenScatter[j * headDim + evenCol] = 1f;
                oddScatter[j * headDim + oddCol] = 1f;
                firstHalfScatter[j * headDim + j] = 1f;
                secondHalfScatter[j * headDim + (half + j)] = 1f;
            }

            return new RopeTables(
                    cos,
                    sin,
                    Tensor.of(cosFlat, Shape.of(ctx, half)),
                    Tensor.of(sinFlat, Shape.of(ctx, half)),
                    Tensor.of(headsOnes, Shape.of(maxHeads, 1)),
                    Tensor.of(evenSelector, Shape.of(headDim, half)),
                    Tensor.of(oddSelector, Shape.of(headDim, half)),
                    Tensor.of(evenScatter, Shape.of(half, headDim)),
                    Tensor.of(oddScatter, Shape.of(half, headDim)),
                    Tensor.of(firstHalfScatter, Shape.of(half, headDim)),
                    Tensor.of(secondHalfScatter, Shape.of(half, headDim)));
        }
    }

    static final class Message implements Llama3ChatFormat.MessageLike {
        final Llama3ChatFormat.Role role;
        final String text;

        Message(Llama3ChatFormat.Role role, String text) {
            this.role = role;
            this.text = text;
        }

        @Override
        public Llama3ChatFormat.Role role() {
            return role;
        }

        @Override
        public String text() {
            return text;
        }
    }
}
