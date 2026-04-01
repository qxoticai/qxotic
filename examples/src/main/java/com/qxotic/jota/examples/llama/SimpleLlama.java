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
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.advanced.Normalizer;
import com.qxotic.toknroll.advanced.Splitter;
import com.qxotic.toknroll.impl.GPT2Tokenizer;
import com.qxotic.toknroll.impl.IntPair;
import com.qxotic.toknroll.impl.RegexSplitter;
import com.qxotic.toknroll.impl.VocabularyImpl;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.LongAdder;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/** Simplified Llama3 inference implementation with optimized prompt processing. */
public final class SimpleLlama {
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

    private static final class TimingProfiler {
        private final boolean enabled;
        private final ConcurrentHashMap<String, LongAdder> nanosByKey = new ConcurrentHashMap<>();
        private final ConcurrentHashMap<String, LongAdder> countByKey = new ConcurrentHashMap<>();

        static final class Snapshot {
            final Map<String, Long> nanosByKey;
            final Map<String, Long> countByKey;

            Snapshot(Map<String, Long> nanosByKey, Map<String, Long> countByKey) {
                this.nanosByKey = nanosByKey;
                this.countByKey = countByKey;
            }
        }

        TimingProfiler(boolean enabled) {
            this.enabled = enabled;
        }

        Scope scope(String key) {
            if (!enabled) {
                return Scope.NOOP;
            }
            return new Scope(this, key, System.nanoTime());
        }

        void reset() {
            nanosByKey.clear();
            countByKey.clear();
        }

        String report(String label) {
            if (!enabled) {
                return "";
            }
            Map<String, Long> nanos = new java.util.HashMap<>();
            Map<String, Long> counts = new java.util.HashMap<>();
            for (Map.Entry<String, LongAdder> entry : nanosByKey.entrySet()) {
                nanos.put(entry.getKey(), entry.getValue().sum());
            }
            for (Map.Entry<String, LongAdder> entry : countByKey.entrySet()) {
                counts.put(entry.getKey(), entry.getValue().sum());
            }
            return renderReport(label, nanos, counts);
        }

        Snapshot snapshot() {
            if (!enabled) {
                return new Snapshot(Map.of(), Map.of());
            }
            Map<String, Long> nanos = new java.util.HashMap<>();
            Map<String, Long> counts = new java.util.HashMap<>();
            for (Map.Entry<String, LongAdder> entry : nanosByKey.entrySet()) {
                nanos.put(entry.getKey(), entry.getValue().sum());
            }
            for (Map.Entry<String, LongAdder> entry : countByKey.entrySet()) {
                counts.put(entry.getKey(), entry.getValue().sum());
            }
            return new Snapshot(nanos, counts);
        }

        String reportDelta(String label, Snapshot since) {
            if (!enabled) {
                return "";
            }
            Map<String, Long> nanosDelta = new java.util.HashMap<>();
            Map<String, Long> countsDelta = new java.util.HashMap<>();

            for (Map.Entry<String, LongAdder> entry : nanosByKey.entrySet()) {
                String key = entry.getKey();
                long now = entry.getValue().sum();
                long prev = since.nanosByKey.getOrDefault(key, 0L);
                long delta = now - prev;
                if (delta > 0L) {
                    nanosDelta.put(key, delta);
                }
            }
            for (Map.Entry<String, LongAdder> entry : countByKey.entrySet()) {
                String key = entry.getKey();
                long now = entry.getValue().sum();
                long prev = since.countByKey.getOrDefault(key, 0L);
                long delta = now - prev;
                if (delta > 0L) {
                    countsDelta.put(key, delta);
                }
            }

            return renderReport(label, nanosDelta, countsDelta);
        }

        private String renderReport(
                String label, Map<String, Long> nanosByKey, Map<String, Long> countByKey) {
            long totalNanos = 0L;
            long gemmNanos = 0L;
            StringBuilder details = new StringBuilder();

            details.append("timings[").append(label).append("]:\n");
            nanosByKey.entrySet().stream()
                    .sorted((a, b) -> Long.compare(b.getValue(), a.getValue()))
                    .forEach(
                            entry -> {
                                String key = entry.getKey();
                                long nanos = entry.getValue();
                                long count = countByKey.getOrDefault(key, 0L);
                                double ms = nanos / 1_000_000.0;
                                double avgUs = count > 0 ? nanos / (double) count / 1_000.0 : 0.0;
                                details.append("  ")
                                        .append(key)
                                        .append(": ")
                                        .append(String.format(java.util.Locale.ROOT, "%.3f ms", ms))
                                        .append(" (count=")
                                        .append(count)
                                        .append(", avg=")
                                        .append(
                                                String.format(
                                                        java.util.Locale.ROOT, "%.3f us", avgUs))
                                        .append(")\n");
                            });

            for (Map.Entry<String, Long> entry : nanosByKey.entrySet()) {
                long nanos = entry.getValue();
                totalNanos += nanos;
                if (entry.getKey().startsWith("gemm.")) {
                    gemmNanos += nanos;
                }
            }
            long restNanos = Math.max(0L, totalNanos - gemmNanos);
            details.append("  gemm.total: ")
                    .append(
                            String.format(
                                    java.util.Locale.ROOT, "%.3f ms", gemmNanos / 1_000_000.0))
                    .append("\n")
                    .append("  rest.total: ")
                    .append(
                            String.format(
                                    java.util.Locale.ROOT, "%.3f ms", restNanos / 1_000_000.0))
                    .append("\n");

            return details.toString();
        }

        private void add(String key, long elapsedNanos) {
            nanosByKey.computeIfAbsent(key, k -> new LongAdder()).add(elapsedNanos);
            countByKey.computeIfAbsent(key, k -> new LongAdder()).increment();
        }

        static final class Scope implements AutoCloseable {
            static final Scope NOOP = new Scope(null, "", 0L);
            private final TimingProfiler profiler;
            private final String key;
            private final long start;

            Scope(TimingProfiler profiler, String key, long start) {
                this.profiler = profiler;
                this.key = key;
                this.start = start;
            }

            @Override
            public void close() {
                if (profiler == null) {
                    return;
                }
                profiler.add(key, System.nanoTime() - start);
            }
        }
    }

    public static void main(String[] args) throws Exception {
        Options options = Options.parse(args);
        Environment env = Environment.withDefaultDevice(DeviceType.PANAMA.deviceIndex(0));
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
        try (LoadedModel loaded = LoadedModel.load(options.modelPath)) {
            if (options.benchmark) {
                runBenchmark(options, loaded);
            } else if (options.interactive) {
                runInteractive(options, loaded);
            } else {
                runInstruct(options, loaded);
            }
        }
    }

    private static void runBenchmark(Options options, LoadedModel loaded) throws IOException {
        LlamaModel model = loaded.model;
        int vocabSize = loaded.config.vocabularySize();
        int measuredRuns = Math.max(1, options.benchmarkRuns);
        int warmupRuns = Math.max(0, options.benchmarkWarmup);
        int totalRuns = warmupRuns + measuredRuns;

        double[] ppResults = new double[measuredRuns];
        double[] tgResults = new double[measuredRuns];

        long modelBytes = Files.size(options.modelPath);
        int threadCount =
                Integer.getInteger(
                        "com.qxotic.llama.kernel.threads",
                        Runtime.getRuntime().availableProcessors());

        System.out.printf(
                "llama-bench style benchmark: model=%s bytes=%.2f GiB backend=CPU threads=%d"
                        + " runs=%d warmup=%d pp=%d tg=%d%n",
                options.modelPath.getFileName(),
                modelBytes / (1024.0 * 1024.0 * 1024.0),
                threadCount,
                measuredRuns,
                warmupRuns,
                options.benchPp,
                options.benchTg);

        for (int run = 0; run < totalRuns; run++) {
            int runSeed = options.benchmarkSeed + run * 9973;
            int[] promptTokens = generateDeterministicTokens(options.benchPp, vocabSize, runSeed);
            int[] decodeTokens =
                    generateDeterministicTokens(options.benchTg, vocabSize, runSeed ^ 0x9E3779B9);

            LlamaState ppState = model.createState();
            model.resetTimings();
            long ppStart = System.nanoTime();
            model.ingestTokens(promptTokens, ppState);
            long ppElapsed = System.nanoTime() - ppStart;
            double ppTps = options.benchPp / (ppElapsed / 1_000_000_000.0);

            LlamaState tgState = model.createState();
            model.resetTimings();
            model.ingestTokens(new int[] {decodeTokens[0]}, tgState);
            long tgStart = System.nanoTime();
            for (int i = 0; i < options.benchTg; i++) {
                model.ingestTokens(new int[] {decodeTokens[i]}, tgState);
                model.computeLogits(tgState);
            }
            long tgElapsed = System.nanoTime() - tgStart;
            double tgTps = options.benchTg / (tgElapsed / 1_000_000_000.0);

            String phase = run < warmupRuns ? "warmup" : "run";
            System.out.printf(
                    "%s %d/%d: pp%d %.2f tok/s, tg%d %.2f tok/s%n",
                    phase, run + 1, totalRuns, options.benchPp, ppTps, options.benchTg, tgTps);

            if (run >= warmupRuns) {
                int idx = run - warmupRuns;
                ppResults[idx] = ppTps;
                tgResults[idx] = tgTps;
            }
        }

        Stats ppStats = summarize(ppResults);
        Stats tgStats = summarize(tgResults);
        System.out.println();
        System.out.printf(
                "| %-30s | %10s | %8s | %7s | %16s | %22s |%n",
                "model", "size", "backend", "threads", "test", "t/s");
        System.out.printf(
                "| %-30s | %10.2f | %8s | %7d | %16s | %10.2f ± %-10.2f |%n",
                "llama all F32",
                modelBytes / (1024.0 * 1024.0 * 1024.0),
                "CPU",
                threadCount,
                "pp" + options.benchPp,
                ppStats.mean,
                ppStats.stddev);
        System.out.printf(
                "| %-30s | %10.2f | %8s | %7d | %16s | %10.2f ± %-10.2f |%n",
                "llama all F32",
                modelBytes / (1024.0 * 1024.0 * 1024.0),
                "CPU",
                threadCount,
                "tg" + options.benchTg,
                tgStats.mean,
                tgStats.stddev);
    }

    private static int[] generateDeterministicTokens(int count, int vocabSize, int seed) {
        int[] tokens = new int[count];
        int x = seed;
        int span = Math.max(1, vocabSize - 3);
        for (int i = 0; i < count; i++) {
            x = x * 1664525 + 1013904223;
            int id = Math.floorMod(x, span) + 2;
            tokens[i] = id;
        }
        return tokens;
    }

    private static Stats summarize(double[] values) {
        double sum = 0.0;
        for (double v : values) {
            sum += v;
        }
        double mean = sum / values.length;
        double var = 0.0;
        for (double v : values) {
            double d = v - mean;
            var += d * d;
        }
        double stddev = Math.sqrt(var / values.length);
        return new Stats(mean, stddev);
    }

    private record Stats(double mean, double stddev) {}

    private static void runInstruct(Options options, LoadedModel loaded) {
        if (options.prompt == null || options.prompt.isBlank()) {
            throw new IllegalArgumentException("--prompt is required in non-interactive mode");
        }
        Llama3ChatFormat format = loaded.chatFormat;
        LlamaModel model = loaded.model;
        Sampler sampler =
                new Sampler(
                        loaded.config.vocabularySize(),
                        options.temperature,
                        options.topP,
                        options.seed);

        LlamaState state = model.createState();

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

        // Batch process all prompt tokens with timing
        int[] promptArray = promptTokens.stream().mapToInt(Integer::intValue).toArray();
        long promptStartNs = System.nanoTime();
        model.resetTimings();
        model.ingestTokens(promptArray, state);
        long promptElapsedNs = System.nanoTime() - promptStartNs;
        double promptTps = promptArray.length / (promptElapsedNs / 1_000_000_000.0);
        String promptTimingReport = model.timingReport("prompt");

        MemoryView<?> logits = model.computeLogits(state);

        IntSequence.Builder generated = IntSequence.newBuilder();
        model.resetTimings();
        long genStartNs = System.nanoTime();
        int decodeTimingBucket =
                Math.max(0, Integer.getInteger("com.qxotic.llama.decode.timing.bucket", 0));
        TimingProfiler.Snapshot bucketSnapshot =
                decodeTimingBucket > 0 ? model.timingSnapshot() : null;
        long bucketStartNs = genStartNs;
        int bucketStartToken = 0;
        int tokenCount = 0;
        for (int i = 0; i < options.maxTokens; i++) {
            int next = sampler.sample(logits);
            generated.add(next);
            tokenCount++;
            if (options.stream) {
                System.out.print(format.stream(IntSequence.of(next)));
            }
            model.ingestTokens(new int[] {next}, state);
            logits = model.computeLogits(state);

            if (decodeTimingBucket > 0 && tokenCount % decodeTimingBucket == 0) {
                long now = System.nanoTime();
                int bucketTokens = tokenCount - bucketStartToken;
                double bucketTps = bucketTokens / ((now - bucketStartNs) / 1_000_000_000.0);
                System.out.printf(
                        "generation bucket %d-%d: %.2f tokens/s (%d)%n",
                        bucketStartToken + 1, tokenCount, bucketTps, bucketTokens);
                String bucketReport =
                        model.timingReportDelta(
                                "generation.bucket." + (bucketStartToken + 1) + "-" + tokenCount,
                                bucketSnapshot);
                if (!bucketReport.isEmpty()) {
                    System.out.print(bucketReport);
                }
                bucketSnapshot = model.timingSnapshot();
                bucketStartNs = now;
                bucketStartToken = tokenCount;
            }

            if (format.stopTokens().contains(next)) {
                break;
            }
        }
        if (decodeTimingBucket > 0 && tokenCount > bucketStartToken) {
            long now = System.nanoTime();
            int bucketTokens = tokenCount - bucketStartToken;
            double bucketTps = bucketTokens / ((now - bucketStartNs) / 1_000_000_000.0);
            System.out.printf(
                    "generation bucket %d-%d: %.2f tokens/s (%d)%n",
                    bucketStartToken + 1, tokenCount, bucketTps, bucketTokens);
            String bucketReport =
                    model.timingReportDelta(
                            "generation.bucket." + (bucketStartToken + 1) + "-" + tokenCount,
                            bucketSnapshot);
            if (!bucketReport.isEmpty()) {
                System.out.print(bucketReport);
            }
        }
        long genElapsedNs = System.nanoTime() - genStartNs;
        double genTps = tokenCount / (genElapsedNs / 1_000_000_000.0);
        String generationTimingReport = model.timingReport("generation");

        if (options.stream) {
            System.out.println();
        } else {
            IntSequence out = generated.build();
            if (!out.isEmpty() && format.stopTokens().contains(out.getLast())) {
                out = out.subSequence(0, out.length() - 1);
            }
            System.out.println(format.echo(out));
        }

        // Print timing statistics
        System.out.printf(
                "context: %d/%d prompt: %.2f tokens/s (%d) generation: %.2f tokens/s (%d)%n",
                state.position,
                loaded.config.contextLength(),
                promptTps,
                promptArray.length,
                genTps,
                tokenCount);
        if (!promptTimingReport.isEmpty()) {
            System.out.print(promptTimingReport);
        }
        if (!generationTimingReport.isEmpty()) {
            System.out.print(generationTimingReport);
        }
    }

    private static void runInteractive(Options options, LoadedModel loaded) {
        Llama3ChatFormat format = loaded.chatFormat;
        LlamaModel model = loaded.model;
        Sampler sampler =
                new Sampler(
                        loaded.config.vocabularySize(),
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

        LlamaState state = model.createState();
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

            // Batch process all new tokens with timing
            int[] newTokens = new int[conversation.size() - consumed];
            for (int i = consumed; i < conversation.size(); i++) {
                newTokens[i - consumed] = conversation.get(i);
            }
            long promptStartNs = System.nanoTime();
            model.resetTimings();
            model.ingestTokens(newTokens, state);
            long promptElapsedNs = System.nanoTime() - promptStartNs;
            double promptTps =
                    newTokens.length > 0
                            ? newTokens.length / (promptElapsedNs / 1_000_000_000.0)
                            : 0.0;
            String promptTimingReport = model.timingReport("prompt");
            consumed = conversation.size();

            MemoryView<?> logits = model.computeLogits(state);
            model.resetTimings();
            long genStartNs = System.nanoTime();
            int decodeTimingBucket =
                    Math.max(0, Integer.getInteger("com.qxotic.llama.decode.timing.bucket", 0));
            TimingProfiler.Snapshot bucketSnapshot =
                    decodeTimingBucket > 0 ? model.timingSnapshot() : null;
            long bucketStartNs = genStartNs;
            int bucketStartToken = 0;
            int tokenCount = 0;
            for (int i = 0; i < options.maxTokens; i++) {
                int next = sampler.sample(logits);
                conversation.add(next);
                consumed++;
                tokenCount++;
                if (options.stream) {
                    System.out.print(format.stream(IntSequence.of(next)));
                }
                model.ingestTokens(new int[] {next}, state);
                logits = model.computeLogits(state);

                if (decodeTimingBucket > 0 && tokenCount % decodeTimingBucket == 0) {
                    long now = System.nanoTime();
                    int bucketTokens = tokenCount - bucketStartToken;
                    double bucketTps = bucketTokens / ((now - bucketStartNs) / 1_000_000_000.0);
                    System.out.printf(
                            "generation bucket %d-%d: %.2f tokens/s (%d)%n",
                            bucketStartToken + 1, tokenCount, bucketTps, bucketTokens);
                    String bucketReport =
                            model.timingReportDelta(
                                    "generation.bucket."
                                            + (bucketStartToken + 1)
                                            + "-"
                                            + tokenCount,
                                    bucketSnapshot);
                    if (!bucketReport.isEmpty()) {
                        System.out.print(bucketReport);
                    }
                    bucketSnapshot = model.timingSnapshot();
                    bucketStartNs = now;
                    bucketStartToken = tokenCount;
                }

                if (format.stopTokens().contains(next)) {
                    break;
                }
            }
            if (decodeTimingBucket > 0 && tokenCount > bucketStartToken) {
                long now = System.nanoTime();
                int bucketTokens = tokenCount - bucketStartToken;
                double bucketTps = bucketTokens / ((now - bucketStartNs) / 1_000_000_000.0);
                System.out.printf(
                        "generation bucket %d-%d: %.2f tokens/s (%d)%n",
                        bucketStartToken + 1, tokenCount, bucketTps, bucketTokens);
                String bucketReport =
                        model.timingReportDelta(
                                "generation.bucket." + (bucketStartToken + 1) + "-" + tokenCount,
                                bucketSnapshot);
                if (!bucketReport.isEmpty()) {
                    System.out.print(bucketReport);
                }
            }
            long genElapsedNs = System.nanoTime() - genStartNs;
            double genTps = tokenCount / (genElapsedNs / 1_000_000_000.0);
            String generationTimingReport = model.timingReport("generation");

            if (options.stream) {
                System.out.println();
            }

            // Print timing statistics
            System.out.printf(
                    "context: %d/%d prompt: %.2f tokens/s (%d) generation: %.2f tokens/s (%d)%n",
                    state.position,
                    loaded.config.contextLength(),
                    promptTps,
                    newTokens.length,
                    genTps,
                    tokenCount);
            if (!promptTimingReport.isEmpty()) {
                System.out.print(promptTimingReport);
            }
            if (!generationTimingReport.isEmpty()) {
                System.out.print(generationTimingReport);
            }
        }
    }

    private static void append(List<Integer> out, IntSequence seq) {
        for (int i = 0; i < seq.length(); i++) {
            out.add(seq.intAt(i));
        }
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

    private record LlamaLayer(
            Tensor attnNorm,
            Tensor wq,
            Tensor wk,
            Tensor wv,
            Tensor wo,
            Tensor ffnNorm,
            Tensor wGate,
            Tensor wDown,
            Tensor wUp) {}

    private record LlamaWeights(
            Tensor tokenTable,
            Tensor output,
            Tensor outputNorm,
            LlamaLayer[] layers,
            RopeTables rope) {}

    private static final class LlamaState {
        final MemoryView<?>[] keyCache;
        final MemoryView<?>[] valueCache;
        final DecodeScratch scratch;
        int position;

        LlamaState(LlamaConfig cfg, MemoryDomain<?> domain) {
            int kvDim = cfg.nKvHeads() * cfg.headDim();
            this.keyCache = new MemoryView<?>[cfg.nLayers()];
            this.valueCache = new MemoryView<?>[cfg.nLayers()];
            for (int i = 0; i < cfg.nLayers(); i++) {
                this.keyCache[i] =
                        MemoryView.of(
                                domain.memoryAllocator()
                                        .allocateMemory(
                                                DataType.FP32,
                                                Shape.of(cfg.contextLength(), kvDim)),
                                DataType.FP32,
                                Layout.rowMajor(Shape.of(cfg.contextLength(), kvDim)));
                this.valueCache[i] =
                        MemoryView.of(
                                domain.memoryAllocator()
                                        .allocateMemory(
                                                DataType.FP32,
                                                Shape.of(kvDim, cfg.contextLength())),
                                DataType.FP32,
                                Layout.rowMajor(Shape.of(kvDim, cfg.contextLength())));
            }
            this.scratch = new DecodeScratch(cfg, domain);
            this.position = 0;
        }
    }

    private static final class DecodeScratch {
        // Single-token buffers
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
        // Batch buffers for layer-wise processing
        final MemoryView<?> batchXBuf;
        final MemoryView<?> batchXTmpBuf;
        final MemoryView<?> batchXbBuf;
        final MemoryView<?> batchQPreBuf;
        final MemoryView<?> batchKPreBuf;
        final MemoryView<?> batchVPreBuf;
        final MemoryView<?> batchFFInBuf;
        final MemoryView<?> batchFfnGateBuf;
        final MemoryView<?> batchFfnUpBuf;
        final MemoryView<?> batchFfnHiddenBuf;
        final MemoryView<?> batchFfnDownBuf;
        final MemoryView<?> batchAttnOutBuf;
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
            int kvDim = cfg.nKvHeads() * cfg.headDim();
            int maxBatch =
                    Math.max(
                            1,
                            Integer.getInteger(
                                    "com.qxotic.llama.prefill.batch",
                                    Integer.getInteger("com.qxotic.llama.batch.size", 512)));
            long xBytes = (long) cfg.dim() * Float.BYTES;
            long xTmpBytes = (long) cfg.dim() * Float.BYTES;
            long xbBytes = (long) cfg.dim() * Float.BYTES;
            long ffInBytes = (long) cfg.dim() * Float.BYTES;
            long qPreBytes = (long) cfg.dim() * Float.BYTES;
            long kPreBytes = (long) kvDim * Float.BYTES;
            long vPreBytes = (long) kvDim * Float.BYTES;
            long qRotBytes = (long) cfg.dim() * Float.BYTES;
            long kRotBytes = (long) kvDim * Float.BYTES;
            long attnScoresBytes = (long) cfg.contextLength() * Float.BYTES;
            long attnOutBytes = (long) cfg.nHeads() * cfg.headDim() * Float.BYTES;
            long ffnGateBytes = (long) cfg.ffnDim() * Float.BYTES;
            long ffnUpBytes = (long) cfg.ffnDim() * Float.BYTES;
            long ffnHiddenBytes = (long) cfg.ffnDim() * Float.BYTES;
            long ffnDownBytes = (long) cfg.dim() * Float.BYTES;
            long logitsBytes = (long) cfg.vocabularySize() * Float.BYTES;
            // Batch buffer sizes
            long batchXBytes = (long) maxBatch * cfg.dim() * Float.BYTES;
            long batchXTmpBytes = (long) maxBatch * cfg.dim() * Float.BYTES;
            long batchXbBytes = (long) maxBatch * cfg.dim() * Float.BYTES;
            long batchQPreBytes = (long) maxBatch * cfg.dim() * Float.BYTES;
            long batchKPreBytes = (long) maxBatch * kvDim * Float.BYTES;
            long batchVPreBytes = (long) maxBatch * kvDim * Float.BYTES;
            long batchFFInBytes = (long) maxBatch * cfg.dim() * Float.BYTES;
            long batchFfnGateBytes = (long) maxBatch * cfg.ffnDim() * Float.BYTES;
            long batchFfnUpBytes = (long) maxBatch * cfg.ffnDim() * Float.BYTES;
            long batchFfnHiddenBytes = (long) maxBatch * cfg.ffnDim() * Float.BYTES;
            long batchFfnDownBytes = (long) maxBatch * cfg.dim() * Float.BYTES;
            long batchAttnOutBytes = (long) maxBatch * cfg.dim() * Float.BYTES;

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
            // Batch buffer offsets
            long offBatchX = alignUp(offBytes, PAGE_BYTES);
            offBytes = offBatchX + batchXBytes;
            long offBatchXTmp = alignUp(offBytes, PAGE_BYTES);
            offBytes = offBatchXTmp + batchXTmpBytes;
            long offBatchXb = alignUp(offBytes, PAGE_BYTES);
            offBytes = offBatchXb + batchXbBytes;
            long offBatchQPre = alignUp(offBytes, PAGE_BYTES);
            offBytes = offBatchQPre + batchQPreBytes;
            long offBatchKPre = alignUp(offBytes, PAGE_BYTES);
            offBytes = offBatchKPre + batchKPreBytes;
            long offBatchVPre = alignUp(offBytes, PAGE_BYTES);
            offBytes = offBatchVPre + batchVPreBytes;
            long offBatchFFIn = alignUp(offBytes, PAGE_BYTES);
            offBytes = offBatchFFIn + batchFFInBytes;
            long offBatchFfnGate = alignUp(offBytes, PAGE_BYTES);
            offBytes = offBatchFfnGate + batchFfnGateBytes;
            long offBatchFfnUp = alignUp(offBytes, PAGE_BYTES);
            offBytes = offBatchFfnUp + batchFfnUpBytes;
            long offBatchFfnHidden = alignUp(offBytes, PAGE_BYTES);
            offBytes = offBatchFfnHidden + batchFfnHiddenBytes;
            long offBatchFfnDown = alignUp(offBytes, PAGE_BYTES);
            offBytes = offBatchFfnDown + batchFfnDownBytes;
            long offBatchAttnOut = alignUp(offBytes, PAGE_BYTES);
            offBytes = offBatchAttnOut + batchAttnOutBytes;

            Memory<?> scratchMemory = domain.memoryAllocator().allocateMemory(offBytes, PAGE_BYTES);
            this.xBuf =
                    MemoryView.of(
                            scratchMemory,
                            offX,
                            DataType.FP32,
                            Layout.rowMajor(Shape.of(1, cfg.dim())));
            this.xTmpBuf =
                    MemoryView.of(
                            scratchMemory,
                            offXTmp,
                            DataType.FP32,
                            Layout.rowMajor(Shape.of(1, cfg.dim())));
            this.xbBuf =
                    MemoryView.of(
                            scratchMemory,
                            offXb,
                            DataType.FP32,
                            Layout.rowMajor(Shape.of(1, cfg.dim())));
            this.ffInBuf =
                    MemoryView.of(
                            scratchMemory,
                            offFfIn,
                            DataType.FP32,
                            Layout.rowMajor(Shape.of(1, cfg.dim())));
            this.qPreBuf =
                    MemoryView.of(
                            scratchMemory,
                            offQPre,
                            DataType.FP32,
                            Layout.rowMajor(Shape.of(1, cfg.dim())));
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
                            Layout.rowMajor(Shape.of(1, cfg.dim())));
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
                            Layout.rowMajor(Shape.of(cfg.contextLength())));
            this.attentionOutBuf =
                    MemoryView.of(
                            scratchMemory,
                            offAttnOut,
                            DataType.FP32,
                            Layout.rowMajor(Shape.of(1, cfg.nHeads() * cfg.headDim())));
            this.ffnGateBuf =
                    MemoryView.of(
                            scratchMemory,
                            offFfnGate,
                            DataType.FP32,
                            Layout.rowMajor(Shape.of(1, cfg.ffnDim())));
            this.ffnUpBuf =
                    MemoryView.of(
                            scratchMemory,
                            offFfnUp,
                            DataType.FP32,
                            Layout.rowMajor(Shape.of(1, cfg.ffnDim())));
            this.ffnHiddenBuf =
                    MemoryView.of(
                            scratchMemory,
                            offFfnHidden,
                            DataType.FP32,
                            Layout.rowMajor(Shape.of(1, cfg.ffnDim())));
            this.ffnDownBuf =
                    MemoryView.of(
                            scratchMemory,
                            offFfnDown,
                            DataType.FP32,
                            Layout.rowMajor(Shape.of(1, cfg.dim())));
            this.logitsBuf =
                    MemoryView.of(
                            scratchMemory,
                            offLogits,
                            DataType.FP32,
                            Layout.rowMajor(Shape.of(cfg.vocabularySize())));
            // Batch buffers
            this.batchXBuf =
                    MemoryView.of(
                            scratchMemory,
                            offBatchX,
                            DataType.FP32,
                            Layout.rowMajor(Shape.of(cfg.dim(), maxBatch)));
            this.batchXTmpBuf =
                    MemoryView.of(
                            scratchMemory,
                            offBatchXTmp,
                            DataType.FP32,
                            Layout.rowMajor(Shape.of(cfg.dim(), maxBatch)));
            this.batchXbBuf =
                    MemoryView.of(
                            scratchMemory,
                            offBatchXb,
                            DataType.FP32,
                            Layout.rowMajor(Shape.of(cfg.dim(), maxBatch)));
            this.batchQPreBuf =
                    MemoryView.of(
                            scratchMemory,
                            offBatchQPre,
                            DataType.FP32,
                            Layout.rowMajor(Shape.of(cfg.dim(), maxBatch)));
            this.batchKPreBuf =
                    MemoryView.of(
                            scratchMemory,
                            offBatchKPre,
                            DataType.FP32,
                            Layout.rowMajor(Shape.of(kvDim, maxBatch)));
            this.batchVPreBuf =
                    MemoryView.of(
                            scratchMemory,
                            offBatchVPre,
                            DataType.FP32,
                            Layout.rowMajor(Shape.of(kvDim, maxBatch)));
            this.batchFFInBuf =
                    MemoryView.of(
                            scratchMemory,
                            offBatchFFIn,
                            DataType.FP32,
                            Layout.rowMajor(Shape.of(cfg.dim(), maxBatch)));
            this.batchFfnGateBuf =
                    MemoryView.of(
                            scratchMemory,
                            offBatchFfnGate,
                            DataType.FP32,
                            Layout.rowMajor(Shape.of(cfg.ffnDim(), maxBatch)));
            this.batchFfnUpBuf =
                    MemoryView.of(
                            scratchMemory,
                            offBatchFfnUp,
                            DataType.FP32,
                            Layout.rowMajor(Shape.of(cfg.ffnDim(), maxBatch)));
            this.batchFfnHiddenBuf =
                    MemoryView.of(
                            scratchMemory,
                            offBatchFfnHidden,
                            DataType.FP32,
                            Layout.rowMajor(Shape.of(cfg.ffnDim(), maxBatch)));
            this.batchFfnDownBuf =
                    MemoryView.of(
                            scratchMemory,
                            offBatchFfnDown,
                            DataType.FP32,
                            Layout.rowMajor(Shape.of(cfg.dim(), maxBatch)));
            this.batchAttnOutBuf =
                    MemoryView.of(
                            scratchMemory,
                            offBatchAttnOut,
                            DataType.FP32,
                            Layout.rowMajor(Shape.of(cfg.dim(), maxBatch)));

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

    private static final class RopeTables {
        final float[][] cos;
        final float[][] sin;

        private RopeTables(float[][] cos, float[][] sin) {
            this.cos = cos;
            this.sin = sin;
        }

        static RopeTables precompute(
                int ctx, int headDim, float theta, float[] ropeScales, int maxHeads) {
            int half = headDim / 2;
            float[][] cos = new float[ctx][half];
            float[][] sin = new float[ctx][half];
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
                }
            }
            return new RopeTables(cos, sin);
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

    private static final class LlamaModel {
        private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
        private static final long PARALLEL_MIN_WORK =
                4096L; // Lowered threshold for better parallelization
        private static final int THREADS =
                Math.max(
                        1,
                        Integer.getInteger(
                                "com.qxotic.llama.kernel.threads",
                                Runtime.getRuntime().availableProcessors()));
        private static final int GEMV_THREADS =
                Math.max(
                        1,
                        Integer.getInteger("com.qxotic.llama.gemv.threads", Math.min(THREADS, 8)));
        private static final int GEMV_PARALLEL_MIN_ROWS =
                Math.max(
                        1,
                        Integer.getInteger(
                                "com.qxotic.llama.gemv.parallel.minRows",
                                Math.max(64, GEMV_THREADS * 8)));
        private static final int LEGACY_BATCH_SIZE =
                Integer.getInteger("com.qxotic.llama.batch.size", 512);
        private static final int PREFILL_BATCH_SIZE =
                Integer.getInteger("com.qxotic.llama.prefill.batch", LEGACY_BATCH_SIZE);
        private static final int PREFILL_UBATCH_SIZE =
                Integer.getInteger(
                        "com.qxotic.llama.prefill.ubatch", Math.min(128, PREFILL_BATCH_SIZE));
        private static final boolean USE_JAVA_WEIGHT_STATIONARY_GEMM =
                Boolean.parseBoolean(System.getProperty("com.qxotic.llama.javawsgemm", "false"));
        private static final int JAVA_WS_TILE_BATCH =
                Integer.getInteger("com.qxotic.llama.javawsgemm.tileBatch", 64);
        private static final int JAVA_WS_TILE_M =
                Integer.getInteger("com.qxotic.llama.javawsgemm.tileM", 32);
        private static final int JAVA_WS_TILE_N =
                Integer.getInteger("com.qxotic.llama.javawsgemm.tileN", 256);
        private static final boolean USE_C_BATCH_GEMM =
                Boolean.parseBoolean(System.getProperty("com.qxotic.llama.cgemm", "true"));
        private static final boolean ENABLE_TIMINGS =
                Boolean.parseBoolean(System.getProperty("com.qxotic.llama.timing", "true"));
        private static final boolean ATTN_VECTOR_SOFTMAX =
                Boolean.parseBoolean(
                        System.getProperty(
                                "com.qxotic.llama.attention.softmax.vectorized", "false"));
        private static final boolean SWIGLU_VECTOR =
                Boolean.parseBoolean(
                        System.getProperty("com.qxotic.llama.swiglu.vectorized", "false"));
        private static final boolean ATTN_DECODE_TWO_PASS =
                Boolean.parseBoolean(
                        System.getProperty("com.qxotic.llama.attention.decode.twopass", "false"));
        private static final boolean ATTN_DECODE_BLOCKED =
                Boolean.parseBoolean(
                        System.getProperty("com.qxotic.llama.attention.decode.blocked", "true"));
        private static final int ATTN_DECODE_BLOCK_SIZE =
                Math.max(
                        16, Integer.getInteger("com.qxotic.llama.attention.decode.blockSize", 128));
        private static final boolean USE_PERSISTENT_PARALLEL =
                Boolean.parseBoolean(System.getProperty("com.qxotic.llama.parallel.pool", "false"));
        private static final long C_BATCH_GEMM_MIN_WORK =
                Long.getLong("com.qxotic.llama.cgemm.minWork", 262_144L);
        private static final int C_BATCH_GEMM_OMP_MIN_WORK =
                Integer.getInteger("com.qxotic.llama.cgemm.omp.minWork", 16_384);
        private static final int C_BATCH_GEMM_OMP_CHUNK =
                Integer.getInteger("com.qxotic.llama.cgemm.omp.chunk", 2);
        private static final int C_BATCH_GEMM_TILE_M =
                Integer.getInteger("com.qxotic.llama.cgemm.tileM", 32);
        private static final int C_BATCH_GEMM_PREFETCH_FLOATS =
                Integer.getInteger("com.qxotic.llama.cgemm.prefetch", 128);
        private static final boolean C_BATCH_GEMM_USE_BLAS =
                Boolean.parseBoolean(System.getProperty("com.qxotic.llama.cgemm.blas", "true"));
        private static final boolean C_BATCH_GEMM_BLAS_REQUIRED =
                Boolean.parseBoolean(
                        System.getProperty("com.qxotic.llama.cgemm.blas.required", "false"));
        private static final long C_BATCH_GEMM_BLAS_MIN_WORK =
                Long.getLong("com.qxotic.llama.cgemm.blas.minWork", 65_536L);
        private static final boolean C_GEMV_USE_BLAS =
                Boolean.parseBoolean(System.getProperty("com.qxotic.llama.cgemv.blas", "false"));
        private static final boolean C_GEMV_BLAS_REQUIRED =
                Boolean.parseBoolean(
                        System.getProperty("com.qxotic.llama.cgemv.blas.required", "false"));
        private static final long C_GEMV_BLAS_MIN_WORK =
                Long.getLong("com.qxotic.llama.cgemv.blas.minWork", 16_384L);
        private static final String C_BATCH_GEMM_KERNEL_NAME = "simple_llama_batch_gemm_omp";
        private static final String C_GEMV_KERNEL_NAME = "simple_llama_gemv_blas";
        private static final String C_BATCH_GEMM_KERNEL_SOURCE =
                buildCBatchGemmKernelSource(
                        C_BATCH_GEMM_OMP_MIN_WORK,
                        C_BATCH_GEMM_OMP_CHUNK,
                        C_BATCH_GEMM_TILE_M,
                        C_BATCH_GEMM_PREFETCH_FLOATS,
                        C_BATCH_GEMM_USE_BLAS,
                        C_BATCH_GEMM_BLAS_REQUIRED,
                        C_BATCH_GEMM_BLAS_MIN_WORK,
                        C_BATCH_GEMM_KERNEL_NAME);
        private static final String C_GEMV_KERNEL_SOURCE =
                buildCGemvKernelSource(
                        C_GEMV_USE_BLAS,
                        C_GEMV_BLAS_REQUIRED,
                        C_GEMV_BLAS_MIN_WORK,
                        C_GEMV_KERNEL_NAME);
        private static final ThreadLocal<GemmScratch> GEMM_SCRATCH =
                ThreadLocal.withInitial(GemmScratch::new);
        private static final ThreadLocal<AttentionScratch> ATTN_SCRATCH =
                ThreadLocal.withInitial(AttentionScratch::new);
        private static final java.util.concurrent.ExecutorService EXEC =
                java.util.concurrent.Executors.newFixedThreadPool(
                        THREADS,
                        r -> {
                            Thread t = new Thread(r, "llama-gemv");
                            t.setDaemon(true);
                            return t;
                        });
        private static final ParallelIndexPool PARALLEL_POOL =
                USE_PERSISTENT_PARALLEL ? new ParallelIndexPool(THREADS, "llama-kernel") : null;

        private static final class ParallelIndexPool {
            private final int parallelism;
            private final int workerCount;
            private final Thread[] workers;
            private final java.util.concurrent.Phaser phaser;
            private final java.util.concurrent.atomic.AtomicReference<Throwable> error =
                    new java.util.concurrent.atomic.AtomicReference<>();
            private volatile java.util.function.IntConsumer task;
            private volatile int count;
            private volatile int chunk;
            private volatile boolean shutdown;

            ParallelIndexPool(int parallelism, String threadPrefix) {
                this.parallelism = Math.max(1, parallelism);
                this.workerCount = Math.max(0, this.parallelism - 1);
                this.phaser = new java.util.concurrent.Phaser(this.workerCount + 1);
                this.workers = new Thread[this.workerCount];
                for (int i = 0; i < this.workerCount; i++) {
                    final int workerId = i + 1;
                    Thread t = new Thread(() -> runWorker(workerId), threadPrefix + "-" + workerId);
                    t.setDaemon(true);
                    t.start();
                    this.workers[i] = t;
                }
            }

            void forRange(int count, int minParallelCount, java.util.function.IntConsumer task) {
                if (count <= 0) {
                    return;
                }
                if (workerCount == 0
                        || count < Math.max(1, minParallelCount)
                        || Thread.currentThread().getName().startsWith("llama-kernel-")) {
                    for (int i = 0; i < count; i++) {
                        task.accept(i);
                    }
                    return;
                }

                this.task = task;
                this.count = count;
                this.chunk = (count + parallelism - 1) / parallelism;
                error.set(null);

                phaser.arriveAndAwaitAdvance();
                runSlice(0);
                phaser.arriveAndAwaitAdvance();

                Throwable t = error.get();
                if (t != null) {
                    if (t instanceof RuntimeException re) {
                        throw re;
                    }
                    if (t instanceof Error e) {
                        throw e;
                    }
                    throw new RuntimeException("Parallel worker failed", t);
                }
            }

            private void runWorker(int workerId) {
                while (true) {
                    phaser.arriveAndAwaitAdvance();
                    if (shutdown) {
                        phaser.arriveAndAwaitAdvance();
                        return;
                    }
                    runSlice(workerId);
                    phaser.arriveAndAwaitAdvance();
                }
            }

            private void runSlice(int workerId) {
                java.util.function.IntConsumer localTask = task;
                int localCount = count;
                int localChunk = chunk;
                int start = workerId * localChunk;
                int end = Math.min(localCount, start + localChunk);
                try {
                    for (int i = start; i < end; i++) {
                        localTask.accept(i);
                    }
                } catch (Throwable t) {
                    error.compareAndSet(null, t);
                }
            }
        }

        private static final class GemmScratch {
            float[] acc = new float[0];
            float[] wVals = new float[0];

            void ensureCapacity(int accSize, int wSize) {
                if (acc.length < accSize) {
                    acc = new float[accSize];
                }
                if (wVals.length < wSize) {
                    wVals = new float[wSize];
                }
            }
        }

        private static final class AttentionScratch {
            float[] q = new float[0];
            float[] scores = new float[0];
            float[] out = new float[0];
            float[] tmp = new float[0];

            void ensureCapacity(int qSize, int scoresSize, int outSize) {
                if (q.length < qSize) {
                    q = new float[qSize];
                }
                if (scores.length < scoresSize) {
                    scores = new float[scoresSize];
                }
                if (out.length < outSize) {
                    out = new float[outSize];
                }
                if (tmp.length < outSize) {
                    tmp = new float[outSize];
                }
            }
        }

        private static final class LayerPlan {
            final Tensor attnNorm;
            final Tensor ffnNorm;
            final MemoryView<?> wq;
            final MemoryView<?> wk;
            final MemoryView<?> wv;
            final MemoryView<?> wo;
            final MemoryView<?> wGate;
            final MemoryView<?> wUp;
            final MemoryView<?> wDown;

            LayerPlan(LlamaLayer layer) {
                this.attnNorm = layer.attnNorm();
                this.ffnNorm = layer.ffnNorm();
                this.wq = layer.wq().materialize();
                this.wk = layer.wk().materialize();
                this.wv = layer.wv().materialize();
                this.wo = layer.wo().materialize();
                this.wGate = layer.wGate().materialize();
                this.wUp = layer.wUp().materialize();
                this.wDown = layer.wDown().materialize();
            }
        }

        private final class DecodeEngine {
            void ingestSingleToken(int token, LlamaState state) {
                DecodeScratch scratch = state.scratch;
                int kvDim = cfg.nKvHeads() * cfg.headDim();

                try (TimingProfiler.Scope ignored = timings.scope("decode.embedding")) {
                    loadEmbeddingIntoScratch(token, scratch);
                }

                for (int layer = 0; layer < cfg.nLayers(); layer++) {
                    LayerPlan plan = layerPlans[layer];
                    attentionLayerStep(state, scratch, layer, plan, kvDim);
                    ffnLayerStep(scratch, plan);
                }

                state.position++;
            }

            private void attentionLayerStep(
                    LlamaState state, DecodeScratch scratch, int layer, LayerPlan plan, int kvDim) {
                try (TimingProfiler.Scope ignored = timings.scope("decode.rmsnorm.attn")) {
                    rmsNormInto(scratch.xTensor, plan.attnNorm, scratch.xbBuf);
                }
                try (TimingProfiler.Scope ignored = timings.scope("decode.proj.qkv")) {
                    projectQkvInto(scratch.xbBuf, plan, scratch);
                }

                Tensor q;
                Tensor k;
                try (TimingProfiler.Scope ignored = timings.scope("decode.rope")) {
                    q =
                            applyRoPETensor(
                                    scratch.qPreTensor,
                                    cfg.nHeads(),
                                    state.position,
                                    w.rope(),
                                    scratch.qRotBuf,
                                    scratch.qRotTensor);
                    k =
                            applyRoPETensor(
                                    scratch.kPreTensor,
                                    cfg.nKvHeads(),
                                    state.position,
                                    w.rope(),
                                    scratch.kRotBuf,
                                    scratch.kRotTensor);
                }

                try (TimingProfiler.Scope ignored = timings.scope("decode.kv.write")) {
                    writeKeyValueCache(state, layer, kvDim, k, scratch.vPreTensor);
                }
                try (TimingProfiler.Scope ignored = timings.scope("decode.attention")) {
                    attention(state, q, layer, scratch);
                }
                try (TimingProfiler.Scope ignored = timings.scope("decode.proj.wo")) {
                    projectInto(
                            scratch.attentionOutBuf,
                            plan.wo,
                            scratch.ffnDownBuf,
                            cfg.dim(),
                            cfg.dim());
                }
                try (TimingProfiler.Scope ignored = timings.scope("decode.residual.attn")) {
                    addInto(scratch.xBuf, scratch.ffnDownBuf, scratch.xTmpBuf, cfg.dim());
                }
            }

            private void ffnLayerStep(DecodeScratch scratch, LayerPlan plan) {
                try (TimingProfiler.Scope ignored = timings.scope("decode.rmsnorm.ffn")) {
                    rmsNormInto(scratch.xTmpTensor, plan.ffnNorm, scratch.ffInBuf);
                }
                try (TimingProfiler.Scope ignored = timings.scope("decode.ffn.gateup")) {
                    projectPairSwigluInto(scratch.ffInBuf, plan, scratch);
                }
                try (TimingProfiler.Scope ignored = timings.scope("decode.ffn.down")) {
                    projectDownInto(
                            scratch.ffnHiddenBuf,
                            plan.wDown,
                            scratch.ffnDownBuf,
                            cfg.dim(),
                            cfg.ffnDim());
                }
                try (TimingProfiler.Scope ignored = timings.scope("decode.residual.ffn")) {
                    addInto(scratch.xTmpBuf, scratch.ffnDownBuf, scratch.xBuf, cfg.dim());
                }
            }
        }

        private final class PrefillEngine {
            void ingestTokens(int[] tokens, LlamaState state) {
                int prefillBatch = Math.max(1, PREFILL_BATCH_SIZE);
                int prefillUbatch = Math.max(1, Math.min(PREFILL_UBATCH_SIZE, prefillBatch));

                for (int chunkStart = 0; chunkStart < tokens.length; chunkStart += prefillBatch) {
                    int chunkSize = Math.min(prefillBatch, tokens.length - chunkStart);
                    for (int ubatchStart = 0;
                            ubatchStart < chunkSize;
                            ubatchStart += prefillUbatch) {
                        int ubatchSize = Math.min(prefillUbatch, chunkSize - ubatchStart);
                        ingestTokensBatched(tokens, chunkStart + ubatchStart, ubatchSize, state);
                    }
                }
            }
        }

        static {
            System.err.println(
                    "[LlamaModel] Using "
                            + THREADS
                            + " threads, prefill batch: "
                            + PREFILL_BATCH_SIZE
                            + ", prefill ubatch: "
                            + PREFILL_UBATCH_SIZE
                            + ", cgemm mode: "
                            + (C_BATCH_GEMM_BLAS_REQUIRED ? "BLAS_REQUIRED" : "BEST_EFFORT")
                            + ", vector species: "
                            + SPECIES.length()
                            + "x"
                            + SPECIES.elementType().getSimpleName());
        }

        private final LlamaConfig cfg;
        private final LlamaWeights w;
        private final DeviceRuntime runtime;
        private final DeviceRuntime cRuntime;
        private final KernelExecutable cBatchGemmKernel;
        private final KernelExecutable cGemvKernel;
        private final ExecutionStream cStream;
        private final MemoryDomain<?> domain;
        private final TimingProfiler timings;
        private final LayerPlan[] layerPlans;
        private final DecodeEngine decodeEngine;
        private final PrefillEngine prefillEngine;

        LlamaModel(LlamaConfig cfg, LlamaWeights w) {
            this.cfg = cfg;
            this.w = w;
            this.runtime = Environment.runtimeFor(Environment.current().defaultDevice());
            DeviceRuntime cRt;
            RuntimeException cRuntimeError = null;
            try {
                cRt = Environment.runtimeFor(DeviceType.C.deviceIndex(0));
            } catch (RuntimeException ex) {
                cRt = null;
                cRuntimeError = ex;
            }
            if (C_BATCH_GEMM_BLAS_REQUIRED && cRt == null) {
                throw new IllegalStateException(
                        "BLAS-required mode enabled but DeviceType.C.deviceIndex(0) runtime is"
                                + " unavailable",
                        cRuntimeError);
            }
            this.cRuntime = cRt;
            this.cBatchGemmKernel = registerCBatchGemmKernel(cRuntime);
            this.cGemvKernel = registerCGemvKernel(cRuntime);
            this.cStream =
                    cRuntime != null ? new ExecutionStream(cRuntime.device(), null, true) : null;
            this.domain = runtime.memoryDomain();
            this.timings = new TimingProfiler(ENABLE_TIMINGS);
            this.layerPlans = new LayerPlan[w.layers().length];
            for (int i = 0; i < w.layers().length; i++) {
                this.layerPlans[i] = new LayerPlan(w.layers()[i]);
            }
            this.decodeEngine = new DecodeEngine();
            this.prefillEngine = new PrefillEngine();
        }

        void resetTimings() {
            timings.reset();
        }

        String timingReport(String label) {
            return timings.report(label);
        }

        TimingProfiler.Snapshot timingSnapshot() {
            return timings.snapshot();
        }

        String timingReportDelta(String label, TimingProfiler.Snapshot since) {
            return timings.reportDelta(label, since);
        }

        private static String buildCBatchGemmKernelSource(
                int ompMinWork,
                int ompChunk,
                int tileM,
                int prefetchFloats,
                boolean useBlas,
                boolean blasRequired,
                long blasMinWork,
                String functionName) {
            int safeOmpMinWork = Math.max(1, ompMinWork);
            int safeChunk = Math.max(1, ompChunk);
            int safeUseBlas = useBlas ? 1 : 0;
            int safeBlasRequired = blasRequired ? 1 : 0;
            long safeBlasMinWork = Math.max(1L, blasMinWork);
            return String.format(
                    java.util.Locale.ROOT,
                    """
                    #include <stdint.h>
                    #include <stddef.h>
                    #include <stdlib.h>
                    #if defined(__linux__) || defined(__APPLE__)
                    #include <dlfcn.h>
                    #define LLAMA_HAS_DLOPEN 1
                    #else
                    #define LLAMA_HAS_DLOPEN 0
                    #endif

                    #define LLAMA_OMP_MIN_WORK %d
                    #define LLAMA_OMP_CHUNK %d
                    #define LLAMA_USE_BLAS %d
                    #define LLAMA_BLAS_REQUIRED %d
                    #define LLAMA_BLAS_MIN_WORK %d

                    enum {
                        LLAMA_CBLAS_ROW_MAJOR = 101,
                        LLAMA_CBLAS_NO_TRANS = 111
                    };

                    typedef void (*llama_cblas_sgemm_fn)(
                            const int,
                            const int,
                            const int,
                            const int,
                            const int,
                            const int,
                            const float,
                            const float *,
                            const int,
                            const float *,
                            const int,
                            const float,
                            float *,
                            const int);

                    static llama_cblas_sgemm_fn llama_cblas_sgemm_ptr = NULL;
                    static int llama_cblas_init_done = 0;

                    static inline void llama_try_init_cblas(void) {
                    #if LLAMA_HAS_DLOPEN
                        if (llama_cblas_init_done) {
                            return;
                        }
                    #if defined(_OPENMP)
                        #pragma omp critical(llama_cblas_init)
                    #endif
                        {
                            if (!llama_cblas_init_done) {
                                void *handle = dlopen("libopenblas.so", RTLD_LAZY | RTLD_LOCAL);
                                if (handle == NULL) {
                                    handle = dlopen("libopenblas.so.0", RTLD_LAZY | RTLD_LOCAL);
                                }
                                if (handle == NULL) {
                                    handle = dlopen("libblas.so.3", RTLD_LAZY | RTLD_LOCAL);
                                }
                                if (handle == NULL) {
                                    handle = dlopen("libblas.so", RTLD_LAZY | RTLD_LOCAL);
                                }
                                if (handle != NULL) {
                                    void *sym = dlsym(handle, "cblas_sgemm");
                                    if (sym != NULL) {
                                        llama_cblas_sgemm_ptr = (llama_cblas_sgemm_fn)sym;
                                    }
                                }
                                llama_cblas_init_done = 1;
                            }
                        }
                    #else
                        llama_cblas_init_done = 1;
                    #endif
                    }

                    static inline int llama_gemm_blas(
                            const float *X,
                            const float *W,
                            float *C,
                            int batch,
                            int m,
                            int n,
                            int xStride,
                            int cStride) {
                    #if LLAMA_USE_BLAS
                        long long work = (long long)batch * (long long)m * (long long)n;
                        if (!LLAMA_BLAS_REQUIRED && work < LLAMA_BLAS_MIN_WORK) {
                            return 0;
                        }
                        if (xStride < batch || cStride < batch) {
                    #if LLAMA_BLAS_REQUIRED
                            abort();
                    #endif
                            return 0;
                        }
                        llama_try_init_cblas();
                        if (llama_cblas_sgemm_ptr == NULL) {
                    #if LLAMA_BLAS_REQUIRED
                            abort();
                    #endif
                            return 0;
                        }
                        llama_cblas_sgemm_ptr(
                                LLAMA_CBLAS_ROW_MAJOR,
                                LLAMA_CBLAS_NO_TRANS,
                                LLAMA_CBLAS_NO_TRANS,
                                m,
                                batch,
                                n,
                                1.0f,
                                W,
                                n,
                                X,
                                xStride,
                                0.0f,
                                C,
                                cStride);
                        return 1;
                    #else
                        (void)X;
                        (void)W;
                        (void)C;
                        (void)batch;
                        (void)m;
                        (void)n;
                        (void)xStride;
                        (void)cStride;
                        return 0;
                    #endif
                    }

                    static inline void llama_gemm_scalar(
                            const float *X,
                            const float *W,
                            float *C,
                            int batch,
                            int m,
                            int n,
                            int xStride,
                            int cStride) {
                    #if defined(_OPENMP)
                        #pragma omp parallel for collapse(2) schedule(static, LLAMA_OMP_CHUNK)
                    #endif
                        for (int row = 0; row < m; row++) {
                            for (int b = 0; b < batch; b++) {
                                const float *wRow = W + (size_t)row * (size_t)n;
                                const float *xCol = X + (size_t)b;
                                float dot = 0.0f;
                                for (int k = 0; k < n; k++) {
                                    dot += wRow[k] * xCol[(size_t)k * (size_t)xStride];
                                }
                                C[(size_t)row * (size_t)cStride + (size_t)b] = dot;
                            }
                        }
                    }

                    void %s(void **buffers, uint64_t *scalars, uint64_t scratch) {
                        (void)scratch;
                        const float *X = (const float *)buffers[0];
                        const float *W = (const float *)buffers[1];
                        float *C = (float *)buffers[2];
                        int batch = (int)scalars[0];
                        int m = (int)scalars[1];
                        int n = (int)scalars[2];
                        int xStride = (int)scalars[3];
                        int cStride = (int)scalars[4];

                        if (X == NULL || W == NULL || C == NULL || batch <= 0 || m <= 0 || n <= 0 || xStride < batch || cStride < batch) {
                    #if LLAMA_BLAS_REQUIRED
                            abort();
                    #endif
                            return;
                        }

                        if (llama_gemm_blas(X, W, C, batch, m, n, xStride, cStride)) {
                            return;
                        }
                    #if LLAMA_BLAS_REQUIRED
                        abort();
                    #endif
                        llama_gemm_scalar(X, W, C, batch, m, n, xStride, cStride);
                    }
                    """,
                    safeOmpMinWork,
                    safeChunk,
                    safeUseBlas,
                    safeBlasRequired,
                    safeBlasMinWork,
                    functionName);
        }

        private static String buildCGemvKernelSource(
                boolean useBlas, boolean blasRequired, long blasMinWork, String functionName) {
            int safeUseBlas = useBlas ? 1 : 0;
            int safeBlasRequired = blasRequired ? 1 : 0;
            long safeBlasMinWork = Math.max(1L, blasMinWork);
            return String.format(
                    java.util.Locale.ROOT,
                    """
                    #include <stdint.h>
                    #include <stddef.h>
                    #include <stdlib.h>
                    #if defined(__linux__) || defined(__APPLE__)
                    #include <dlfcn.h>
                    #define LLAMA_HAS_DLOPEN 1
                    #else
                    #define LLAMA_HAS_DLOPEN 0
                    #endif

                    #define LLAMA_USE_BLAS %d
                    #define LLAMA_BLAS_REQUIRED %d
                    #define LLAMA_BLAS_MIN_WORK %d

                    enum {
                        LLAMA_CBLAS_ROW_MAJOR = 101,
                        LLAMA_CBLAS_NO_TRANS = 111
                    };

                    typedef void (*llama_cblas_sgemv_fn)(
                            const int,
                            const int,
                            const int,
                            const int,
                            const float,
                            const float *,
                            const int,
                            const float *,
                            const int,
                            const float,
                            float *,
                            const int);

                    static llama_cblas_sgemv_fn llama_cblas_sgemv_ptr = NULL;
                    static int llama_cblas_init_done = 0;

                    static inline void llama_try_init_cblas(void) {
                    #if LLAMA_HAS_DLOPEN
                        if (llama_cblas_init_done) {
                            return;
                        }
                    #if defined(_OPENMP)
                        #pragma omp critical(llama_cblas_gemv_init)
                    #endif
                        {
                            if (!llama_cblas_init_done) {
                                void *handle = dlopen("libopenblas.so", RTLD_LAZY | RTLD_LOCAL);
                                if (handle == NULL) {
                                    handle = dlopen("libopenblas.so.0", RTLD_LAZY | RTLD_LOCAL);
                                }
                                if (handle == NULL) {
                                    handle = dlopen("libblas.so.3", RTLD_LAZY | RTLD_LOCAL);
                                }
                                if (handle == NULL) {
                                    handle = dlopen("libblas.so", RTLD_LAZY | RTLD_LOCAL);
                                }
                                if (handle != NULL) {
                                    void *sym = dlsym(handle, "cblas_sgemv");
                                    if (sym != NULL) {
                                        llama_cblas_sgemv_ptr = (llama_cblas_sgemv_fn)sym;
                                    }
                                }
                                llama_cblas_init_done = 1;
                            }
                        }
                    #else
                        llama_cblas_init_done = 1;
                    #endif
                    }

                    static inline int llama_gemv_blas(
                            const float *A,
                            const float *X,
                            float *Y,
                            int m,
                            int n) {
                    #if LLAMA_USE_BLAS
                        long long work = (long long)m * (long long)n;
                        if (!LLAMA_BLAS_REQUIRED && work < LLAMA_BLAS_MIN_WORK) {
                            return 0;
                        }
                        if (A == NULL || X == NULL || Y == NULL || m <= 0 || n <= 0) {
                    #if LLAMA_BLAS_REQUIRED
                            abort();
                    #endif
                            return 0;
                        }
                        llama_try_init_cblas();
                        if (llama_cblas_sgemv_ptr == NULL) {
                    #if LLAMA_BLAS_REQUIRED
                            abort();
                    #endif
                            return 0;
                        }
                        llama_cblas_sgemv_ptr(
                                LLAMA_CBLAS_ROW_MAJOR,
                                LLAMA_CBLAS_NO_TRANS,
                                m,
                                n,
                                1.0f,
                                A,
                                n,
                                X,
                                1,
                                0.0f,
                                Y,
                                1);
                        return 1;
                    #else
                        (void)A;
                        (void)X;
                        (void)Y;
                        (void)m;
                        (void)n;
                        return 0;
                    #endif
                    }

                    static inline void llama_gemv_scalar(
                            const float *A,
                            const float *X,
                            float *Y,
                            int m,
                            int n) {
                        for (int row = 0; row < m; row++) {
                            float dot = 0.0f;
                            const float *aRow = A + (size_t)row * (size_t)n;
                            for (int col = 0; col < n; col++) {
                                dot += aRow[col] * X[col];
                            }
                            Y[row] = dot;
                        }
                    }

                    void %s(void **buffers, uint64_t *scalars, uint64_t scratch) {
                        (void)scratch;
                        const float *A = (const float *)buffers[0];
                        const float *X = (const float *)buffers[1];
                        float *Y = (float *)buffers[2];
                        int m = (int)scalars[0];
                        int n = (int)scalars[1];

                        if (A == NULL || X == NULL || Y == NULL || m <= 0 || n <= 0) {
                    #if LLAMA_BLAS_REQUIRED
                            abort();
                    #endif
                            return;
                        }

                        if (llama_gemv_blas(A, X, Y, m, n)) {
                            return;
                        }
                    #if LLAMA_BLAS_REQUIRED
                        abort();
                    #endif
                        llama_gemv_scalar(A, X, Y, m, n);
                    }
                    """,
                    safeUseBlas,
                    safeBlasRequired,
                    safeBlasMinWork,
                    functionName);
        }

        private static KernelExecutable registerCBatchGemmKernel(DeviceRuntime runtime) {
            if (runtime == null || !USE_C_BATCH_GEMM) {
                return null;
            }
            if (!runtime.supportsKernels()) {
                if (C_BATCH_GEMM_BLAS_REQUIRED) {
                    throw new IllegalStateException(
                            "BLAS-required mode enabled but C runtime does not support kernels");
                }
                return null;
            }
            try {
                return runtime.loadRegisteredExecutable(C_BATCH_GEMM_KERNEL_NAME)
                        .orElseGet(
                                () ->
                                        runtime.registerKernel(
                                                C_BATCH_GEMM_KERNEL_NAME,
                                                KernelProgram.source(
                                                        "c",
                                                        C_BATCH_GEMM_KERNEL_SOURCE,
                                                        C_BATCH_GEMM_KERNEL_NAME)));
            } catch (RuntimeException ex) {
                if (C_BATCH_GEMM_BLAS_REQUIRED) {
                    throw new IllegalStateException(
                            "BLAS-required mode enabled but C GEMM kernel registration failed", ex);
                }
                return null;
            }
        }

        private static KernelExecutable registerCGemvKernel(DeviceRuntime runtime) {
            if (runtime == null || !USE_C_BATCH_GEMM || !C_GEMV_USE_BLAS) {
                return null;
            }
            if (!runtime.supportsKernels()) {
                if (C_GEMV_BLAS_REQUIRED) {
                    throw new IllegalStateException(
                            "BLAS-required mode enabled but C runtime does not support kernels");
                }
                return null;
            }
            try {
                return runtime.loadRegisteredExecutable(C_GEMV_KERNEL_NAME)
                        .orElseGet(
                                () ->
                                        runtime.registerKernel(
                                                C_GEMV_KERNEL_NAME,
                                                KernelProgram.source(
                                                        "c",
                                                        C_GEMV_KERNEL_SOURCE,
                                                        C_GEMV_KERNEL_NAME)));
            } catch (RuntimeException ex) {
                if (C_GEMV_BLAS_REQUIRED) {
                    throw new IllegalStateException(
                            "BLAS-required mode enabled but C GEMV kernel registration failed", ex);
                }
                return null;
            }
        }

        private static boolean shouldUseCBatchGemm(int batchSize, int m, int n) {
            if (!USE_C_BATCH_GEMM) {
                return false;
            }
            if (C_BATCH_GEMM_BLAS_REQUIRED) {
                return true;
            }
            long work = (long) batchSize * (long) m * (long) n;
            return work >= C_BATCH_GEMM_MIN_WORK;
        }

        private static boolean shouldUseCGemv(int m, int n) {
            if (!USE_C_BATCH_GEMM || !C_GEMV_USE_BLAS) {
                return false;
            }
            if (C_GEMV_BLAS_REQUIRED) {
                return true;
            }
            long work = (long) m * (long) n;
            return work >= C_GEMV_BLAS_MIN_WORK;
        }

        LlamaState createState() {
            return new LlamaState(cfg, domain);
        }

        void ingestTokens(int[] tokens, LlamaState state) {
            if (tokens.length == 0) {
                return;
            }

            if (tokens.length == 1) {
                try (TimingProfiler.Scope ignored = timings.scope("op.ingest.single")) {
                    decodeEngine.ingestSingleToken(tokens[0], state);
                }
                return;
            }

            prefillEngine.ingestTokens(tokens, state);
        }

        private void ingestSingleToken(int token, LlamaState state) {
            decodeEngine.ingestSingleToken(token, state);
        }

        private void ingestTokensBatched(int[] tokens, LlamaState state) {
            ingestTokensBatched(tokens, 0, tokens.length, state);
        }

        private void ingestTokensBatched(
                int[] tokens, int tokenOffset, int batchSize, LlamaState state) {
            DecodeScratch scratch = state.scratch;
            int kvDim = cfg.nKvHeads() * cfg.headDim();

            // Step 1: Load embeddings for all tokens into batch buffer (parallelized)
            try (TimingProfiler.Scope ignored = timings.scope("op.embedding.batch")) {
                parallelForBatch(
                        batchSize,
                        i -> {
                            loadEmbeddingIntoBatchBuffer(
                                    tokens[tokenOffset + i], scratch.batchXBuf, i);
                        });
            }

            // Step 2: Process all tokens layer by layer
            for (int layer = 0; layer < cfg.nLayers(); layer++) {
                final int currentLayer = layer; // Make final for lambda capture
                final LayerPlan plan = layerPlans[currentLayer];

                // Attention path
                // 2a: Batch RMSNorm
                try (TimingProfiler.Scope ignored = timings.scope("op.rmsnorm.attn")) {
                    rmsNormBatch(scratch.batchXBuf, plan.attnNorm, scratch.batchXbBuf, batchSize);
                }

                // 2b: Batch QKV projection
                try (TimingProfiler.Scope ignored = timings.scope("gemm.q")) {
                    batchGemm(
                            scratch.batchXbBuf,
                            plan.wq,
                            scratch.batchQPreBuf,
                            batchSize,
                            cfg.dim(),
                            cfg.dim());
                }
                try (TimingProfiler.Scope ignored = timings.scope("gemm.k")) {
                    batchGemm(
                            scratch.batchXbBuf,
                            plan.wk,
                            scratch.batchKPreBuf,
                            batchSize,
                            kvDim,
                            cfg.dim());
                }
                try (TimingProfiler.Scope ignored = timings.scope("gemm.v")) {
                    batchGemm(
                            scratch.batchXbBuf,
                            plan.wv,
                            scratch.batchVPreBuf,
                            batchSize,
                            kvDim,
                            cfg.dim());
                }

                // 2c: Apply RoPE and write KV cache in parallel
                try (TimingProfiler.Scope ignored = timings.scope("op.rope.kv")) {
                    parallelForBatch(
                            batchSize,
                            i -> {
                                int position = state.position + i;

                                // Apply RoPE in-place in batch buffers
                                applyRoPEInPlace(
                                        scratch.batchQPreBuf,
                                        i,
                                        cfg.nHeads(),
                                        position,
                                        w.rope(),
                                        cfg.dim());
                                applyRoPEInPlace(
                                        scratch.batchKPreBuf,
                                        i,
                                        cfg.nKvHeads(),
                                        position,
                                        w.rope(),
                                        kvDim);

                                // Write to KV cache directly from batch buffer
                                writeKeyValueCacheFromBatch(
                                        state,
                                        currentLayer,
                                        kvDim,
                                        scratch.batchKPreBuf,
                                        scratch.batchVPreBuf,
                                        i,
                                        position);
                            });
                }

                // 2d: Batch attention with causal masking
                try (TimingProfiler.Scope ignored = timings.scope("op.attention.batch")) {
                    batchAttention(state, currentLayer, scratch, batchSize);
                }

                // 2e: Batch project attention output
                try (TimingProfiler.Scope ignored = timings.scope("gemm.wo")) {
                    batchGemm(
                            scratch.batchAttnOutBuf,
                            plan.wo,
                            scratch.batchFfnDownBuf,
                            batchSize,
                            cfg.dim(),
                            cfg.dim());
                }

                // 2f: Add residual in parallel
                try (TimingProfiler.Scope ignored = timings.scope("op.residual.attn")) {
                    parallelForBatch(
                            batchSize,
                            i -> {
                                addIntoBatch(
                                        scratch.batchXBuf,
                                        scratch.batchFfnDownBuf,
                                        scratch.batchXTmpBuf,
                                        i,
                                        cfg.dim());
                            });
                }

                // FFN path
                // 2g: Batch RMSNorm on attention output
                try (TimingProfiler.Scope ignored = timings.scope("op.rmsnorm.ffn")) {
                    rmsNormBatch(
                            scratch.batchXTmpBuf, plan.ffnNorm, scratch.batchFFInBuf, batchSize);
                }

                // 2h: Batch FFN projections (Gate and Up)
                try (TimingProfiler.Scope ignored = timings.scope("gemm.gate")) {
                    batchGemm(
                            scratch.batchFFInBuf,
                            plan.wGate,
                            scratch.batchFfnGateBuf,
                            batchSize,
                            cfg.ffnDim(),
                            cfg.dim());
                }
                try (TimingProfiler.Scope ignored = timings.scope("gemm.up")) {
                    batchGemm(
                            scratch.batchFFInBuf,
                            plan.wUp,
                            scratch.batchFfnUpBuf,
                            batchSize,
                            cfg.ffnDim(),
                            cfg.dim());
                }

                // 2i: SwiGLU activation on batch
                try (TimingProfiler.Scope ignored = timings.scope("op.swiglu")) {
                    swigluBatch(
                            scratch.batchFfnGateBuf,
                            scratch.batchFfnUpBuf,
                            scratch.batchFfnHiddenBuf,
                            batchSize,
                            cfg.ffnDim());
                }

                // 2j: Batch FFN down projection
                try (TimingProfiler.Scope ignored = timings.scope("gemm.down")) {
                    batchGemm(
                            scratch.batchFfnHiddenBuf,
                            plan.wDown,
                            scratch.batchFfnDownBuf,
                            batchSize,
                            cfg.dim(),
                            cfg.ffnDim());
                }

                // 2k: Add residual in parallel
                try (TimingProfiler.Scope ignored = timings.scope("op.residual.ffn")) {
                    parallelForBatch(
                            batchSize,
                            i -> {
                                addIntoBatch(
                                        scratch.batchXTmpBuf,
                                        scratch.batchFfnDownBuf,
                                        scratch.batchXBuf,
                                        i,
                                        cfg.dim());
                            });
                }
            }

            // Step 3: Copy final output of last token to single-token buffer for logits
            copyFromBatch(scratch.batchXBuf, batchSize - 1, scratch.xBuf, cfg.dim());
            state.position += batchSize;
        }

        private void loadEmbeddingIntoBatchBuffer(int token, MemoryView<?> batchBuf, int index) {
            MemoryView<?> tokenTable = w.tokenTable().materialize();
            @SuppressWarnings({"rawtypes", "unchecked"})
            MemoryAccess access =
                    Environment.runtimeFor(tokenTable.memory().device())
                            .memoryDomain()
                            .directAccess();
            int dim = cfg.dim();
            int batchStride = PREFILL_BATCH_SIZE;
            long tokenOffset = tokenTable.byteOffset() + (long) token * dim * Float.BYTES;
            long batchBase = batchBuf.byteOffset();

            for (int i = 0; i < dim; i++) {
                float v =
                        access.readFloat(tokenTable.memory(), tokenOffset + (long) i * Float.BYTES);
                long batchOff = batchBase + ((long) i * batchStride + index) * Float.BYTES;
                access.writeFloat(batchBuf.memory(), batchOff, v);
            }
        }

        private void copyFromBatch(
                MemoryView<?> batchBuf, int index, MemoryView<?> singleBuf, int n) {
            @SuppressWarnings({"rawtypes", "unchecked"})
            MemoryAccess access =
                    Environment.runtimeFor(batchBuf.memory().device())
                            .memoryDomain()
                            .directAccess();
            int batchStride = PREFILL_BATCH_SIZE;
            long batchBase = batchBuf.byteOffset();
            long singleOffset = singleBuf.byteOffset();
            for (int i = 0; i < n; i++) {
                float v =
                        access.readFloat(
                                batchBuf.memory(),
                                batchBase + ((long) i * batchStride + index) * Float.BYTES);
                access.writeFloat(singleBuf.memory(), singleOffset + (long) i * Float.BYTES, v);
            }
        }

        private void copyToBatch(
                MemoryView<?> singleBuf, int index, MemoryView<?> batchBuf, int n) {
            @SuppressWarnings({"rawtypes", "unchecked"})
            MemoryAccess access =
                    Environment.runtimeFor(singleBuf.memory().device())
                            .memoryDomain()
                            .directAccess();
            long singleOffset = singleBuf.byteOffset();
            long batchOffset = batchBuf.byteOffset() + (long) index * n * Float.BYTES;
            for (int i = 0; i < n; i++) {
                float v =
                        access.readFloat(singleBuf.memory(), singleOffset + (long) i * Float.BYTES);
                access.writeFloat(batchBuf.memory(), batchOffset + (long) i * Float.BYTES, v);
            }
        }

        private void rmsNormBatch(
                MemoryView<?> input, Tensor weight, MemoryView<?> output, int batchSize) {
            int dim = cfg.dim();

            // Parallelize for larger batches
            if (batchSize >= 4) {
                parallelForBatch(
                        batchSize,
                        i -> {
                            rmsNormBatchElement(input, weight, output, i, dim);
                        });
            } else {
                for (int i = 0; i < batchSize; i++) {
                    rmsNormBatchElement(input, weight, output, i, dim);
                }
            }
        }

        private void rmsNormBatchElement(
                MemoryView<?> input, Tensor weight, MemoryView<?> output, int batchIndex, int dim) {
            MemoryView<?> wv = weight.materialize();
            @SuppressWarnings({"rawtypes", "unchecked"})
            MemoryAccess access =
                    Environment.runtimeFor(input.memory().device()).memoryDomain().directAccess();
            int batchStride = PREFILL_BATCH_SIZE;

            long inBase = input.byteOffset();
            long outBase = output.byteOffset();
            long wBase = wv.byteOffset();

            // Compute sum of squares
            double sumSq = 0.0;
            for (int i = 0; i < dim; i++) {
                long off = inBase + ((long) i * batchStride + batchIndex) * Float.BYTES;
                float v = access.readFloat(input.memory(), off);
                sumSq += v * v;
            }

            float invRms = (float) (1.0 / Math.sqrt(sumSq / dim + cfg.rmsEps()));

            for (int i = 0; i < dim; i++) {
                long inOff = inBase + ((long) i * batchStride + batchIndex) * Float.BYTES;
                long outOff = outBase + ((long) i * batchStride + batchIndex) * Float.BYTES;
                float xVal = access.readFloat(input.memory(), inOff);
                float wVal = access.readFloat(wv.memory(), wBase + (long) i * Float.BYTES);
                access.writeFloat(output.memory(), outOff, xVal * invRms * wVal);
            }
        }

        private void swigluBatch(
                MemoryView<?> gateBuf,
                MemoryView<?> upBuf,
                MemoryView<?> outBuf,
                int batchSize,
                int ffnDim) {
            @SuppressWarnings({"rawtypes", "unchecked"})
            MemoryAccess access =
                    Environment.runtimeFor(gateBuf.memory().device()).memoryDomain().directAccess();
            MemorySegment gateSeg = ((MemoryView<MemorySegment>) gateBuf).memory().base();
            MemorySegment upSeg = ((MemoryView<MemorySegment>) upBuf).memory().base();
            MemorySegment outSeg = ((MemoryView<MemorySegment>) outBuf).memory().base();

            int batchStride = PREFILL_BATCH_SIZE;
            long gateBase = gateBuf.byteOffset();
            long upBase = upBuf.byteOffset();
            long outBase = outBuf.byteOffset();
            int vecLen = SPECIES.length();
            int vecBound = SPECIES.loopBound(batchSize);
            FloatVector oneVec = FloatVector.broadcast(SPECIES, 1.0f);

            for (int f = 0; f < ffnDim; f++) {
                long rowBase = (long) f * batchStride;
                if (SWIGLU_VECTOR) {
                    int b = 0;
                    for (; b < vecBound; b += vecLen) {
                        long gateOff = gateBase + (rowBase + b) * Float.BYTES;
                        long upOff = upBase + (rowBase + b) * Float.BYTES;
                        long outOff = outBase + (rowBase + b) * Float.BYTES;

                        FloatVector gate =
                                FloatVector.fromMemorySegment(
                                        SPECIES, gateSeg, gateOff, ByteOrder.nativeOrder());
                        FloatVector up =
                                FloatVector.fromMemorySegment(
                                        SPECIES, upSeg, upOff, ByteOrder.nativeOrder());
                        FloatVector silu =
                                gate.div(gate.neg().lanewise(VectorOperators.EXP).add(oneVec));
                        silu.mul(up).intoMemorySegment(outSeg, outOff, ByteOrder.nativeOrder());
                    }

                    for (; b < batchSize; b++) {
                        long gateOff = gateBase + (rowBase + b) * Float.BYTES;
                        long upOff = upBase + (rowBase + b) * Float.BYTES;
                        long outOff = outBase + (rowBase + b) * Float.BYTES;
                        float gate = access.readFloat(gateBuf.memory(), gateOff);
                        float up = access.readFloat(upBuf.memory(), upOff);
                        float silu = gate / (1.0f + (float) Math.exp(-gate));
                        access.writeFloat(outBuf.memory(), outOff, silu * up);
                    }
                } else {
                    for (int b = 0; b < batchSize; b++) {
                        long gateOff = gateBase + (rowBase + b) * Float.BYTES;
                        long upOff = upBase + (rowBase + b) * Float.BYTES;
                        long outOff = outBase + (rowBase + b) * Float.BYTES;
                        float gate = access.readFloat(gateBuf.memory(), gateOff);
                        float up = access.readFloat(upBuf.memory(), upOff);
                        float silu = gate / (1.0f + (float) Math.exp(-gate));
                        access.writeFloat(outBuf.memory(), outOff, silu * up);
                    }
                }
            }
        }

        private void batchGemm(
                MemoryView<?> A, MemoryView<?> B, MemoryView<?> C, int batchSize, int m, int n) {
            // Prefill convention: X is column-batch [n, batchSize], W is [m, n], output is [m,
            // batchSize].
            // This computes Y = W * X.

            if (C_BATCH_GEMM_BLAS_REQUIRED && (cBatchGemmKernel == null || cStream == null)) {
                throw new IllegalStateException(
                        "BLAS-required mode enabled but C GEMM kernel is unavailable");
            }

            if (cBatchGemmKernel != null
                    && cStream != null
                    && shouldUseCBatchGemm(batchSize, m, n)) {
                cBatchGemmKernel.launch(
                        LaunchConfig.auto(),
                        KernelArgs.fromVarargs(
                                A, B, C, batchSize, m, n, PREFILL_BATCH_SIZE, PREFILL_BATCH_SIZE),
                        cStream);
                return;
            }

            if (USE_JAVA_WEIGHT_STATIONARY_GEMM) {
                batchGemmWeightStationaryJava(A, B, C, batchSize, m, n);
                return;
            }

            @SuppressWarnings({"rawtypes", "unchecked"})
            MemoryAccess access =
                    Environment.runtimeFor(A.memory().device()).memoryDomain().directAccess();

            MemorySegment aSeg = ((MemoryView<MemorySegment>) A).memory().base();
            MemorySegment bSeg = ((MemoryView<MemorySegment>) B).memory().base();
            MemorySegment cSeg = ((MemoryView<MemorySegment>) C).memory().base();

            long aBase = A.byteOffset();
            long bBase = B.byteOffset();
            long cBase = C.byteOffset();

            int vecLen = SPECIES.length();
            int vecBound = SPECIES.loopBound(n);

            // Process each batch element in parallel
            parallelForBatch(
                    batchSize,
                    b -> {
                        long aRowBase = aBase + (long) b * n * Float.BYTES;
                        long cRowBase = cBase + (long) b * m * Float.BYTES;

                        // For each output element in this batch
                        for (int row = 0; row < m; row++) {
                            long bRowBase = bBase + (long) row * n * Float.BYTES;

                            FloatVector acc1 = FloatVector.zero(SPECIES);
                            FloatVector acc2 = FloatVector.zero(SPECIES);
                            FloatVector acc3 = FloatVector.zero(SPECIES);
                            FloatVector acc4 = FloatVector.zero(SPECIES);

                            int col = 0;
                            int unrollBound = (n / (vecLen * 4)) * (vecLen * 4);

                            for (; col < unrollBound; col += vecLen * 4) {
                                long aOff1 = aRowBase + (long) col * Float.BYTES;
                                long bOff1 = bRowBase + (long) col * Float.BYTES;
                                FloatVector av1 =
                                        FloatVector.fromMemorySegment(
                                                SPECIES, aSeg, aOff1, ByteOrder.nativeOrder());
                                FloatVector bv1 =
                                        FloatVector.fromMemorySegment(
                                                SPECIES, bSeg, bOff1, ByteOrder.nativeOrder());
                                acc1 = av1.fma(bv1, acc1);

                                long aOff2 = aOff1 + (long) vecLen * Float.BYTES;
                                long bOff2 = bOff1 + (long) vecLen * Float.BYTES;
                                FloatVector av2 =
                                        FloatVector.fromMemorySegment(
                                                SPECIES, aSeg, aOff2, ByteOrder.nativeOrder());
                                FloatVector bv2 =
                                        FloatVector.fromMemorySegment(
                                                SPECIES, bSeg, bOff2, ByteOrder.nativeOrder());
                                acc2 = av2.fma(bv2, acc2);

                                long aOff3 = aOff2 + (long) vecLen * Float.BYTES;
                                long bOff3 = bOff2 + (long) vecLen * Float.BYTES;
                                FloatVector av3 =
                                        FloatVector.fromMemorySegment(
                                                SPECIES, aSeg, aOff3, ByteOrder.nativeOrder());
                                FloatVector bv3 =
                                        FloatVector.fromMemorySegment(
                                                SPECIES, bSeg, bOff3, ByteOrder.nativeOrder());
                                acc3 = av3.fma(bv3, acc3);

                                long aOff4 = aOff3 + (long) vecLen * Float.BYTES;
                                long bOff4 = bOff3 + (long) vecLen * Float.BYTES;
                                FloatVector av4 =
                                        FloatVector.fromMemorySegment(
                                                SPECIES, aSeg, aOff4, ByteOrder.nativeOrder());
                                FloatVector bv4 =
                                        FloatVector.fromMemorySegment(
                                                SPECIES, bSeg, bOff4, ByteOrder.nativeOrder());
                                acc4 = av4.fma(bv4, acc4);
                            }

                            for (; col < vecBound; col += vecLen) {
                                long aOff = aRowBase + (long) col * Float.BYTES;
                                long bOff = bRowBase + (long) col * Float.BYTES;
                                FloatVector av =
                                        FloatVector.fromMemorySegment(
                                                SPECIES, aSeg, aOff, ByteOrder.nativeOrder());
                                FloatVector bv =
                                        FloatVector.fromMemorySegment(
                                                SPECIES, bSeg, bOff, ByteOrder.nativeOrder());
                                acc1 = av.fma(bv, acc1);
                            }

                            float dot =
                                    acc1.add(acc2)
                                            .add(acc3)
                                            .add(acc4)
                                            .reduceLanes(VectorOperators.ADD);

                            for (; col < n; col++) {
                                float av =
                                        access.readFloat(
                                                A.memory(), aRowBase + (long) col * Float.BYTES);
                                float bv =
                                        access.readFloat(
                                                B.memory(), bRowBase + (long) col * Float.BYTES);
                                dot += av * bv;
                            }

                            access.writeFloat(C.memory(), cRowBase + (long) row * Float.BYTES, dot);
                        }
                    });
        }

        @SuppressWarnings({"rawtypes", "unchecked"})
        private void batchGemmWeightStationaryJava(
                MemoryView<?> A, MemoryView<?> B, MemoryView<?> C, int batchSize, int m, int n) {
            MemoryAccess access =
                    Environment.runtimeFor(A.memory().device()).memoryDomain().directAccess();
            MemorySegment aSeg = ((MemoryView<MemorySegment>) A).memory().base();

            long aBase = A.byteOffset();
            long bBase = B.byteOffset();
            long cBase = C.byteOffset();
            int aLd = PREFILL_BATCH_SIZE;
            int cLd = PREFILL_BATCH_SIZE;

            int tileM = Math.max(1, JAVA_WS_TILE_M);
            int tileN = Math.max(1, JAVA_WS_TILE_N);
            int vecLen = SPECIES.length();
            int vecBatchBound = SPECIES.loopBound(batchSize);

            int mBlocks = (m + tileM - 1) / tileM;
            long work = (long) batchSize * m * n;

            java.util.function.BiConsumer<Integer, Integer> blockKernel =
                    (bb, mb) -> {
                        int mStart = mb * tileM;
                        int mEnd = Math.min(m, mStart + tileM);
                        int mCount = mEnd - mStart;
                        GemmScratch scratch = GEMM_SCRATCH.get();
                        scratch.ensureCapacity(batchSize * mCount, mCount);
                        float[] acc = scratch.acc;
                        float[] wVals = scratch.wVals;

                        Arrays.fill(acc, 0, batchSize * mCount, 0.0f);

                        for (int kStart = 0; kStart < n; kStart += tileN) {
                            int kEnd = Math.min(n, kStart + tileN);
                            for (int k = kStart; k < kEnd; k++) {
                                for (int rowLocal = 0; rowLocal < mCount; rowLocal++) {
                                    int row = mStart + rowLocal;
                                    long wOff = bBase + ((long) row * n + k) * Float.BYTES;
                                    wVals[rowLocal] = access.readFloat(B.memory(), wOff);
                                }

                                long xBase = aBase + (long) k * aLd * Float.BYTES;
                                int b = 0;
                                for (; b < vecBatchBound; b += vecLen) {
                                    FloatVector xVec =
                                            FloatVector.fromMemorySegment(
                                                    SPECIES,
                                                    aSeg,
                                                    xBase + (long) b * Float.BYTES,
                                                    ByteOrder.nativeOrder());
                                    for (int rowLocal = 0; rowLocal < mCount; rowLocal++) {
                                        int accBase = rowLocal * batchSize + b;
                                        FloatVector accVec =
                                                FloatVector.fromArray(SPECIES, acc, accBase);
                                        FloatVector wVec =
                                                FloatVector.broadcast(SPECIES, wVals[rowLocal]);
                                        xVec.fma(wVec, accVec).intoArray(acc, accBase);
                                    }
                                }
                                for (; b < batchSize; b++) {
                                    float xv =
                                            access.readFloat(
                                                    A.memory(), xBase + (long) b * Float.BYTES);
                                    for (int rowLocal = 0; rowLocal < mCount; rowLocal++) {
                                        acc[rowLocal * batchSize + b] += wVals[rowLocal] * xv;
                                    }
                                }
                            }
                        }

                        for (int rowLocal = 0; rowLocal < mCount; rowLocal++) {
                            int row = mStart + rowLocal;
                            int accBase = rowLocal * batchSize;
                            for (int b = 0; b < batchSize; b++) {
                                long cOff = cBase + ((long) row * cLd + b) * Float.BYTES;
                                access.writeFloat(C.memory(), cOff, acc[accBase + b]);
                            }
                        }
                    };

            if (work < PARALLEL_MIN_WORK || THREADS <= 1 || mBlocks <= 1) {
                for (int bb = 0; bb < 1; bb++) {
                    for (int mb = 0; mb < mBlocks; mb++) {
                        blockKernel.accept(bb, mb);
                    }
                }
                return;
            }
            parallelForBatched(1, mBlocks, blockKernel);
        }

        @SuppressWarnings({"rawtypes", "unchecked"})
        private void batchGemmTiled(
                MemoryView<?> A,
                MemoryView<?> B,
                MemoryView<?> C,
                MemorySegment aSeg,
                MemorySegment bSeg,
                MemorySegment cSeg,
                long aBase,
                long bBase,
                long cBase,
                MemoryAccess access,
                int batchSize,
                int m,
                int n,
                int bStart,
                int bEnd,
                int tileM,
                int tileN,
                int tileB) {

            int vecLen = SPECIES.length();

            // Process batch in tiles
            for (int bb = bStart; bb < bEnd; bb += tileB) {
                int bBlockEnd = Math.min(bEnd, bb + tileB);

                // Process rows in tiles (cache-friendly for B matrix)
                for (int mm = 0; mm < m; mm += tileM) {
                    int mBlockEnd = Math.min(m, mm + tileM);

                    // Process columns in tiles
                    for (int nn = 0; nn < n; nn += tileN) {
                        int nBlockEnd = Math.min(n, nn + tileN);
                        int nTileSize = nBlockEnd - nn;
                        int nVecBound = nn + SPECIES.loopBound(nTileSize);

                        // Innermost loops: compute tile
                        for (int b = bb; b < bBlockEnd; b++) {
                            long aRowBase = aBase + (long) b * n * Float.BYTES;
                            long cRowBase = cBase + (long) b * m * Float.BYTES;

                            for (int row = mm; row < mBlockEnd; row++) {
                                long bRowBase = bBase + (long) row * n * Float.BYTES;

                                // Accumulate dot product for this tile
                                FloatVector acc1 = FloatVector.zero(SPECIES);
                                FloatVector acc2 = FloatVector.zero(SPECIES);
                                FloatVector acc3 = FloatVector.zero(SPECIES);
                                FloatVector acc4 = FloatVector.zero(SPECIES);

                                int col = nn;

                                // Unroll by 4 for better ILP
                                int unrollBound = nn + ((nTileSize / (vecLen * 4)) * (vecLen * 4));
                                for (; col < unrollBound; col += vecLen * 4) {
                                    long aOff1 = aRowBase + (long) col * Float.BYTES;
                                    long bOff1 = bRowBase + (long) col * Float.BYTES;

                                    FloatVector av1 =
                                            FloatVector.fromMemorySegment(
                                                    SPECIES, aSeg, aOff1, ByteOrder.nativeOrder());
                                    FloatVector bv1 =
                                            FloatVector.fromMemorySegment(
                                                    SPECIES, bSeg, bOff1, ByteOrder.nativeOrder());
                                    acc1 = av1.fma(bv1, acc1);

                                    long aOff2 = aOff1 + (long) vecLen * Float.BYTES;
                                    long bOff2 = bOff1 + (long) vecLen * Float.BYTES;
                                    FloatVector av2 =
                                            FloatVector.fromMemorySegment(
                                                    SPECIES, aSeg, aOff2, ByteOrder.nativeOrder());
                                    FloatVector bv2 =
                                            FloatVector.fromMemorySegment(
                                                    SPECIES, bSeg, bOff2, ByteOrder.nativeOrder());
                                    acc2 = av2.fma(bv2, acc2);

                                    long aOff3 = aOff2 + (long) vecLen * Float.BYTES;
                                    long bOff3 = bOff2 + (long) vecLen * Float.BYTES;
                                    FloatVector av3 =
                                            FloatVector.fromMemorySegment(
                                                    SPECIES, aSeg, aOff3, ByteOrder.nativeOrder());
                                    FloatVector bv3 =
                                            FloatVector.fromMemorySegment(
                                                    SPECIES, bSeg, bOff3, ByteOrder.nativeOrder());
                                    acc3 = av3.fma(bv3, acc3);

                                    long aOff4 = aOff3 + (long) vecLen * Float.BYTES;
                                    long bOff4 = bOff3 + (long) vecLen * Float.BYTES;
                                    FloatVector av4 =
                                            FloatVector.fromMemorySegment(
                                                    SPECIES, aSeg, aOff4, ByteOrder.nativeOrder());
                                    FloatVector bv4 =
                                            FloatVector.fromMemorySegment(
                                                    SPECIES, bSeg, bOff4, ByteOrder.nativeOrder());
                                    acc4 = av4.fma(bv4, acc4);
                                }

                                // Handle remaining vectors
                                for (; col < nVecBound; col += vecLen) {
                                    long aOff = aRowBase + (long) col * Float.BYTES;
                                    long bOff = bRowBase + (long) col * Float.BYTES;
                                    FloatVector av =
                                            FloatVector.fromMemorySegment(
                                                    SPECIES, aSeg, aOff, ByteOrder.nativeOrder());
                                    FloatVector bv =
                                            FloatVector.fromMemorySegment(
                                                    SPECIES, bSeg, bOff, ByteOrder.nativeOrder());
                                    acc1 = av.fma(bv, acc1);
                                }

                                // Combine accumulators
                                float dot =
                                        acc1.add(acc2)
                                                .add(acc3)
                                                .add(acc4)
                                                .reduceLanes(VectorOperators.ADD);

                                // Scalar tail
                                for (; col < nBlockEnd; col++) {
                                    float av =
                                            access.readFloat(
                                                    A.memory(),
                                                    aRowBase + (long) col * Float.BYTES);
                                    float bv =
                                            access.readFloat(
                                                    B.memory(),
                                                    bRowBase + (long) col * Float.BYTES);
                                    dot += av * bv;
                                }

                                // Accumulate to output (handle first tile vs subsequent tiles)
                                long cOff = cRowBase + (long) row * Float.BYTES;
                                if (nn == 0) {
                                    access.writeFloat(C.memory(), cOff, dot);
                                } else {
                                    float prev = access.readFloat(C.memory(), cOff);
                                    access.writeFloat(C.memory(), cOff, prev + dot);
                                }
                            }
                        }
                    }
                }
            }
        }

        private void parallelForBatched(
                int batchSize, int rows, java.util.function.BiConsumer<Integer, Integer> body) {
            int total = batchSize * rows;
            if (total <= 0) {
                return;
            }
            if (THREADS <= 1 || total < 16) {
                for (int b = 0; b < batchSize; b++) {
                    for (int r = 0; r < rows; r++) {
                        body.accept(b, r);
                    }
                }
                return;
            }
            if (USE_PERSISTENT_PARALLEL) {
                PARALLEL_POOL.forRange(
                        total,
                        16,
                        idx -> {
                            int b = idx / rows;
                            int r = idx % rows;
                            body.accept(b, r);
                        });
                return;
            }
            int workers = Math.min(THREADS, total);
            int chunk = (total + workers - 1) / workers;
            java.util.List<java.util.concurrent.Future<?>> futures =
                    new java.util.ArrayList<>(workers);
            for (int w = 0; w < workers; w++) {
                int start = w * chunk;
                int end = Math.min(total, start + chunk);
                if (start >= end) continue;
                futures.add(
                        EXEC.submit(
                                () -> {
                                    for (int idx = start; idx < end; idx++) {
                                        int b = idx / rows;
                                        int r = idx % rows;
                                        body.accept(b, r);
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

        private void parallelForBatch(int batchSize, java.util.function.IntConsumer body) {
            // Don't parallelize for small batches (overhead not worth it)
            if (batchSize < 4) {
                for (int i = 0; i < batchSize; i++) {
                    body.accept(i);
                }
                return;
            }
            if (USE_PERSISTENT_PARALLEL) {
                PARALLEL_POOL.forRange(batchSize, 4, body);
                return;
            }

            int workers = Math.min(THREADS, batchSize);
            if (workers <= 1) {
                for (int i = 0; i < batchSize; i++) {
                    body.accept(i);
                }
                return;
            }
            int chunk = (batchSize + workers - 1) / workers;
            java.util.List<java.util.concurrent.Future<?>> futures =
                    new java.util.ArrayList<>(workers);
            for (int w = 0; w < workers; w++) {
                int start = w * chunk;
                int end = Math.min(batchSize, start + chunk);
                if (start >= end) continue;
                futures.add(
                        EXEC.submit(
                                () -> {
                                    for (int i = start; i < end; i++) {
                                        body.accept(i);
                                    }
                                }));
            }
            for (java.util.concurrent.Future<?> future : futures) {
                try {
                    future.get();
                } catch (InterruptedException ex) {
                    Thread.currentThread().interrupt();
                    throw new RuntimeException("Batch worker interrupted", ex);
                } catch (java.util.concurrent.ExecutionException ex) {
                    throw new RuntimeException("Batch worker failed", ex.getCause());
                }
            }
        }

        private void applyRoPEInPlace(
                MemoryView<?> batchBuf,
                int batchIndex,
                int nHeads,
                int position,
                RopeTables rope,
                int dim) {
            @SuppressWarnings({"rawtypes", "unchecked"})
            MemoryAccess access =
                    Environment.runtimeFor(batchBuf.memory().device())
                            .memoryDomain()
                            .directAccess();
            int headDim = cfg.headDim();
            int batchStride = PREFILL_BATCH_SIZE;
            long baseOffset = batchBuf.byteOffset();
            float[] cos = rope.cos[position];
            float[] sin = rope.sin[position];
            int half = headDim / 2;

            for (int head = 0; head < nHeads; head++) {
                int base = head * headDim;
                for (int i = 0; i < half; i++) {
                    int even = base + (i << 1);
                    int odd = even + 1;
                    long evenOff =
                            baseOffset + ((long) even * batchStride + batchIndex) * Float.BYTES;
                    long oddOff =
                            baseOffset + ((long) odd * batchStride + batchIndex) * Float.BYTES;

                    float x0 = access.readFloat(batchBuf.memory(), evenOff);
                    float x1 = access.readFloat(batchBuf.memory(), oddOff);
                    float c = cos[i];
                    float s = sin[i];

                    access.writeFloat(batchBuf.memory(), evenOff, x0 * c - x1 * s);
                    access.writeFloat(batchBuf.memory(), oddOff, x0 * s + x1 * c);
                }
            }
        }

        private void writeKeyValueCacheFromBatch(
                LlamaState state,
                int layer,
                int kvDim,
                MemoryView<?> batchKBuf,
                MemoryView<?> batchVBuf,
                int batchIndex,
                int position) {
            @SuppressWarnings({"rawtypes", "unchecked"})
            MemoryAccess access =
                    Environment.runtimeFor(batchKBuf.memory().device())
                            .memoryDomain()
                            .directAccess();

            MemoryView<?> keyCacheView = state.keyCache[layer];
            MemoryView<?> valueCacheView = state.valueCache[layer];

            int batchStride = PREFILL_BATCH_SIZE;
            long kBatchBase = batchKBuf.byteOffset();
            long vBatchBase = batchVBuf.byteOffset();
            long keyCacheBase = keyCacheView.byteOffset();
            long valueCacheBase = valueCacheView.byteOffset();
            long keyRowOffset = (long) position * kvDim * Float.BYTES;

            for (int i = 0; i < kvDim; i++) {
                float kVal =
                        access.readFloat(
                                batchKBuf.memory(),
                                kBatchBase + ((long) i * batchStride + batchIndex) * Float.BYTES);
                float vVal =
                        access.readFloat(
                                batchVBuf.memory(),
                                vBatchBase + ((long) i * batchStride + batchIndex) * Float.BYTES);

                access.writeFloat(
                        keyCacheView.memory(),
                        keyCacheBase + keyRowOffset + (long) i * Float.BYTES,
                        kVal);
                access.writeFloat(
                        valueCacheView.memory(),
                        valueCacheBase
                                + (long) i * cfg.contextLength() * Float.BYTES
                                + (long) position * Float.BYTES,
                        vVal);
            }
        }

        private void batchAttention(
                LlamaState state, int layer, DecodeScratch scratch, int batchSize) {
            // Compute attention for all tokens in parallel with causal masking
            // For each token i, it attends to positions [0, state.position + i]

            int kvDim = cfg.nKvHeads() * cfg.headDim();
            int headDim = cfg.headDim();
            int kvMul = cfg.nHeads() / cfg.nKvHeads();
            float invScale = 1f / (float) Math.sqrt(headDim);

            MemoryView<?> qBuf = scratch.batchQPreBuf; // Already has RoPE applied
            MemoryView<?> keyCache = state.keyCache[layer];
            MemoryView<?> valueCache = state.valueCache[layer];
            MemoryView<?> outBuf = scratch.batchAttnOutBuf;

            @SuppressWarnings({"rawtypes", "unchecked"})
            MemoryAccess access =
                    Environment.runtimeFor(qBuf.memory().device()).memoryDomain().directAccess();
            MemorySegment keySeg = ((MemoryView<MemorySegment>) keyCache).memory().base();
            MemorySegment valueSeg = ((MemoryView<MemorySegment>) valueCache).memory().base();

            long qBase = qBuf.byteOffset();
            long keyBase = keyCache.byteOffset();
            long valueBase = valueCache.byteOffset();
            long outBase = outBuf.byteOffset();
            int batchStride = PREFILL_BATCH_SIZE;
            long keyRowStrideBytes = (long) kvDim * Float.BYTES;
            long valueRowStrideBytes = (long) cfg.contextLength() * Float.BYTES;
            int vecLen = SPECIES.length();
            int vecBoundHead = SPECIES.loopBound(headDim);

            // Parallelize over batch and heads
            parallelForBatched(
                    batchSize,
                    cfg.nHeads(),
                    (b, head) -> {
                        int position = state.position + b;
                        int length = position + 1;

                        int kvHead = head / kvMul;
                        int kvHeadOffset = kvHead * headDim;
                        int headOffset = head * headDim;

                        long qHeadBase =
                                qBase + ((long) headOffset * batchStride + b) * Float.BYTES;
                        long outHeadBase =
                                outBase + ((long) headOffset * batchStride + b) * Float.BYTES;

                        AttentionScratch local = ATTN_SCRATCH.get();
                        local.ensureCapacity(headDim, length, headDim);
                        float[] qHead = local.q;
                        float[] scores = local.scores;

                        for (int d = 0; d < headDim; d++) {
                            qHead[d] =
                                    access.readFloat(
                                            qBuf.memory(),
                                            qHeadBase + (long) d * batchStride * Float.BYTES);
                        }

                        // Compute attention scores: Q @ K^T
                        float maxScore = Float.NEGATIVE_INFINITY;
                        for (int t = 0; t < length; t++) {
                            long kHeadBase =
                                    keyBase
                                            + (long) t * keyRowStrideBytes
                                            + (long) kvHeadOffset * Float.BYTES;
                            FloatVector accVec = FloatVector.zero(SPECIES);
                            int d = 0;
                            for (; d < vecBoundHead; d += vecLen) {
                                FloatVector qVec = FloatVector.fromArray(SPECIES, qHead, d);
                                FloatVector kVec =
                                        FloatVector.fromMemorySegment(
                                                SPECIES,
                                                keySeg,
                                                kHeadBase + (long) d * Float.BYTES,
                                                ByteOrder.nativeOrder());
                                accVec = qVec.fma(kVec, accVec);
                            }
                            float dot = accVec.reduceLanes(VectorOperators.ADD);
                            for (; d < headDim; d++) {
                                float kv =
                                        access.readFloat(
                                                keyCache.memory(),
                                                kHeadBase + (long) d * Float.BYTES);
                                dot += qHead[d] * kv;
                            }
                            float scaled = dot * invScale;
                            scores[t] = scaled;
                            if (scaled > maxScore) maxScore = scaled;
                        }

                        // Softmax
                        int vecBoundLen = SPECIES.loopBound(length);
                        int t = 0;
                        if (ATTN_VECTOR_SOFTMAX) {
                            FloatVector maxVec =
                                    FloatVector.broadcast(SPECIES, Float.NEGATIVE_INFINITY);
                            for (; t < vecBoundLen; t += vecLen) {
                                FloatVector sVec = FloatVector.fromArray(SPECIES, scores, t);
                                maxVec = sVec.max(maxVec);
                            }
                            maxScore = Math.max(maxScore, maxVec.reduceLanes(VectorOperators.MAX));
                            for (; t < length; t++) {
                                if (scores[t] > maxScore) {
                                    maxScore = scores[t];
                                }
                            }
                            double sumExp = 0d;
                            FloatVector maxScoreVec = FloatVector.broadcast(SPECIES, maxScore);
                            t = 0;
                            for (; t < vecBoundLen; t += vecLen) {
                                FloatVector sVec = FloatVector.fromArray(SPECIES, scores, t);
                                FloatVector eVec =
                                        sVec.sub(maxScoreVec).lanewise(VectorOperators.EXP);
                                eVec.intoArray(scores, t);
                                sumExp += eVec.reduceLanes(VectorOperators.ADD);
                            }
                            for (; t < length; t++) {
                                float e = (float) Math.exp(scores[t] - maxScore);
                                scores[t] = e;
                                sumExp += e;
                            }

                            float invSum = (float) (1.0d / sumExp);
                            FloatVector invSumVec = FloatVector.broadcast(SPECIES, invSum);
                            t = 0;
                            for (; t < vecBoundLen; t += vecLen) {
                                FloatVector sVec = FloatVector.fromArray(SPECIES, scores, t);
                                sVec.mul(invSumVec).intoArray(scores, t);
                            }
                            for (; t < length; t++) {
                                scores[t] *= invSum;
                            }
                        } else {
                            double sumExp = 0d;
                            for (t = 0; t < length; t++) {
                                float e = (float) Math.exp(scores[t] - maxScore);
                                scores[t] = e;
                                sumExp += e;
                            }
                            float invSum = (float) (1.0d / sumExp);
                            for (t = 0; t < length; t++) {
                                scores[t] *= invSum;
                            }
                        }

                        // Weighted sum: scores @ V
                        for (int d = 0; d < headDim; d++) {
                            long valueRowBase =
                                    valueBase + (long) (kvHeadOffset + d) * valueRowStrideBytes;
                            FloatVector accValVec = FloatVector.zero(SPECIES);
                            t = 0;
                            for (; t < vecBoundLen; t += vecLen) {
                                FloatVector pVec = FloatVector.fromArray(SPECIES, scores, t);
                                FloatVector vVec =
                                        FloatVector.fromMemorySegment(
                                                SPECIES,
                                                valueSeg,
                                                valueRowBase + (long) t * Float.BYTES,
                                                ByteOrder.nativeOrder());
                                accValVec = pVec.fma(vVec, accValVec);
                            }
                            float accVal = accValVec.reduceLanes(VectorOperators.ADD);
                            for (; t < length; t++) {
                                float p = scores[t];
                                float vv =
                                        access.readFloat(
                                                valueCache.memory(),
                                                valueRowBase + (long) t * Float.BYTES);
                                accVal += p * vv;
                            }
                            access.writeFloat(
                                    outBuf.memory(),
                                    outHeadBase + (long) d * batchStride * Float.BYTES,
                                    accVal);
                        }
                    });
        }

        private void addIntoBatch(
                MemoryView<?> aBatch,
                MemoryView<?> bBatch,
                MemoryView<?> outBatch,
                int batchIndex,
                int n) {
            @SuppressWarnings({"rawtypes", "unchecked"})
            MemoryAccess access =
                    Environment.runtimeFor(aBatch.memory().device()).memoryDomain().directAccess();

            int batchStride = PREFILL_BATCH_SIZE;
            long aBase = aBatch.byteOffset();
            long bBase = bBatch.byteOffset();
            long outBase = outBatch.byteOffset();

            for (int i = 0; i < n; i++) {
                long off = ((long) i * batchStride + batchIndex) * Float.BYTES;
                float av = access.readFloat(aBatch.memory(), aBase + off);
                float bv = access.readFloat(bBatch.memory(), bBase + off);
                access.writeFloat(outBatch.memory(), outBase + off, av + bv);
            }
        }

        MemoryView<?> computeLogits(LlamaState state) {
            DecodeScratch scratch = state.scratch;
            rmsNormInto(scratch.xTensor, w.outputNorm(), scratch.ffInBuf);
            projectInto(state, scratch.ffInTensor, w.output(), scratch.logitsBuf);
            return scratch.logitsBuf;
        }

        private void loadEmbeddingIntoScratch(int token, DecodeScratch scratch) {
            MemoryView<?> tokenTable = w.tokenTable().materialize();
            @SuppressWarnings({"rawtypes", "unchecked"})
            MemoryAccess access =
                    Environment.runtimeFor(tokenTable.memory().device())
                            .memoryDomain()
                            .directAccess();
            int dim = cfg.dim();
            long tokenOffset = tokenTable.byteOffset() + (long) token * dim * Float.BYTES;
            for (int i = 0; i < dim; i++) {
                float v =
                        access.readFloat(tokenTable.memory(), tokenOffset + (long) i * Float.BYTES);
                access.writeFloat(
                        scratch.xBuf.memory(),
                        scratch.xBuf.byteOffset() + (long) i * Float.BYTES,
                        v);
            }
        }

        private void rmsNormInto(Tensor x, Tensor weight, MemoryView<?> out) {
            MemoryView<?> xv = x.materialize();
            MemoryView<?> wv = weight.materialize();
            @SuppressWarnings({"rawtypes", "unchecked"})
            MemoryAccess access =
                    Environment.runtimeFor(xv.memory().device()).memoryDomain().directAccess();
            int dim = cfg.dim();
            long xBase = xv.byteOffset();
            long wBase = wv.byteOffset();
            long outBase = out.byteOffset();
            double sumSq = 0.0;
            for (int i = 0; i < dim; i++) {
                float v = access.readFloat(xv.memory(), xBase + (long) i * Float.BYTES);
                sumSq += v * v;
            }
            float invRms = (float) (1.0 / Math.sqrt(sumSq / dim + cfg.rmsEps()));
            for (int i = 0; i < dim; i++) {
                float xVal = access.readFloat(xv.memory(), xBase + (long) i * Float.BYTES);
                float wVal = access.readFloat(wv.memory(), wBase + (long) i * Float.BYTES);
                access.writeFloat(
                        out.memory(), outBase + (long) i * Float.BYTES, xVal * invRms * wVal);
            }
        }

        private void projectInto(
                LlamaState state, Tensor x, Tensor weight, MemoryView<?> outBuffer) {
            MemoryView<?> weightView = weight.materialize();
            MemoryView<?> xView = x.materialize();

            if (x.dataType() != DataType.FP32 || weight.dataType() != DataType.FP32) {
                throw new IllegalStateException("projectInto requires FP32 tensors");
            }

            int m = (int) weight.shape().size(0);
            int n = (int) weight.shape().size(1);

            gemv(weightView, xView, outBuffer, m, n);
        }

        private void projectQkvInto(
                LlamaState state,
                Tensor x,
                Tensor wq,
                Tensor wk,
                Tensor wv,
                DecodeScratch scratch) {
            MemoryView<?> xView = x.materialize();
            MemoryView<?> wqView = wq.materialize();
            MemoryView<?> wkView = wk.materialize();
            MemoryView<?> wvView = wv.materialize();

            int dim = cfg.dim();
            int kvDim = cfg.nKvHeads() * cfg.headDim();

            gemv(wqView, xView, scratch.qPreBuf, dim, dim);
            gemv(wkView, xView, scratch.kPreBuf, kvDim, dim);
            gemv(wvView, xView, scratch.vPreBuf, kvDim, dim);
        }

        private void projectQkvInto(MemoryView<?> xView, LayerPlan plan, DecodeScratch scratch) {
            int dim = cfg.dim();
            int kvDim = cfg.nKvHeads() * cfg.headDim();

            gemv(plan.wq, xView, scratch.qPreBuf, dim, dim);
            gemv(plan.wk, xView, scratch.kPreBuf, kvDim, dim);
            gemv(plan.wv, xView, scratch.vPreBuf, kvDim, dim);
        }

        private void projectInto(
                MemoryView<?> xView,
                MemoryView<?> weightView,
                MemoryView<?> outBuffer,
                int m,
                int n) {
            gemv(weightView, xView, outBuffer, m, n);
        }

        private void projectDownInto(
                LlamaState state, Tensor x, Tensor wDown, MemoryView<?> outBuffer) {
            MemoryView<?> weightView = wDown.materialize();
            MemoryView<?> xView = x.materialize();

            int m = (int) wDown.shape().size(0);
            int n = (int) wDown.shape().size(1);

            gemv(weightView, xView, outBuffer, m, n);
        }

        private void projectDownInto(
                MemoryView<?> xView,
                MemoryView<?> wDownView,
                MemoryView<?> outBuffer,
                int m,
                int n) {
            gemv(wDownView, xView, outBuffer, m, n);
        }

        private void projectPairSwigluInto(
                LlamaState state, Tensor x, Tensor wGate, Tensor wUp, DecodeScratch scratch) {
            MemoryView<?> xView = x.materialize();
            MemoryView<?> gateView = wGate.materialize();
            MemoryView<?> upView = wUp.materialize();

            int m = (int) wGate.shape().size(0);
            int n = (int) wGate.shape().size(1);

            gemv(gateView, xView, scratch.ffnGateBuf, m, n);
            gemv(upView, xView, scratch.ffnUpBuf, m, n);
            applySwigluFromScratch(scratch, m);
        }

        private void projectPairSwigluInto(
                MemoryView<?> xView, LayerPlan plan, DecodeScratch scratch) {
            int m = cfg.ffnDim();
            int n = cfg.dim();

            gemv(plan.wGate, xView, scratch.ffnGateBuf, m, n);
            gemv(plan.wUp, xView, scratch.ffnUpBuf, m, n);

            applySwigluFromScratch(scratch, m);
        }

        private void applySwigluFromScratch(DecodeScratch scratch, int m) {

            @SuppressWarnings({"rawtypes", "unchecked"})
            MemoryAccess access =
                    Environment.runtimeFor(scratch.ffnGateBuf.memory().device())
                            .memoryDomain()
                            .directAccess();

            long gateBase = scratch.ffnGateBuf.byteOffset();
            long upBase = scratch.ffnUpBuf.byteOffset();
            long outBase = scratch.ffnHiddenBuf.byteOffset();

            MemorySegment gateSeg =
                    ((MemoryView<MemorySegment>) scratch.ffnGateBuf).memory().base();
            MemorySegment upSeg = ((MemoryView<MemorySegment>) scratch.ffnUpBuf).memory().base();
            MemorySegment outSeg =
                    ((MemoryView<MemorySegment>) scratch.ffnHiddenBuf).memory().base();

            int upper = SPECIES.loopBound(m);
            int i = 0;
            for (; i < upper; i += SPECIES.length()) {
                long gateOff = gateBase + (long) i * Float.BYTES;
                long upOff = upBase + (long) i * Float.BYTES;
                long outOff = outBase + (long) i * Float.BYTES;

                FloatVector gate =
                        FloatVector.fromMemorySegment(
                                SPECIES, gateSeg, gateOff, ByteOrder.nativeOrder());
                FloatVector up =
                        FloatVector.fromMemorySegment(
                                SPECIES, upSeg, upOff, ByteOrder.nativeOrder());

                FloatVector one = FloatVector.broadcast(SPECIES, 1.0f);
                FloatVector negGate = gate.neg();
                FloatVector expNegGate = negGate.lanewise(VectorOperators.EXP);
                FloatVector denom = expNegGate.add(one);
                FloatVector silu = gate.div(denom);

                FloatVector result = silu.mul(up);
                result.intoMemorySegment(outSeg, outOff, ByteOrder.nativeOrder());
            }

            for (; i < m; i++) {
                float gate =
                        access.readFloat(
                                scratch.ffnGateBuf.memory(), gateBase + (long) i * Float.BYTES);
                float up =
                        access.readFloat(
                                scratch.ffnUpBuf.memory(), upBase + (long) i * Float.BYTES);
                float silu = gate / (1.0f + (float) Math.exp(-gate));
                access.writeFloat(
                        scratch.ffnHiddenBuf.memory(), outBase + (long) i * Float.BYTES, silu * up);
            }
        }

        private void gemv(MemoryView<?> A, MemoryView<?> x, MemoryView<?> y, int m, int n) {
            // Optimized matrix-vector multiplication: y = A @ x
            // A: [m, n], x: [n], y: [m]

            if (C_GEMV_BLAS_REQUIRED && (cGemvKernel == null || cStream == null)) {
                throw new IllegalStateException(
                        "BLAS-required mode enabled but C GEMV kernel is unavailable");
            }
            if (cGemvKernel != null && cStream != null && shouldUseCGemv(m, n)) {
                cGemvKernel.launch(
                        LaunchConfig.auto(), KernelArgs.fromVarargs(A, x, y, m, n), cStream);
                return;
            }

            @SuppressWarnings({"rawtypes", "unchecked"})
            MemoryAccess access =
                    Environment.runtimeFor(A.memory().device()).memoryDomain().directAccess();

            MemorySegment aSeg = ((MemoryView<MemorySegment>) A).memory().base();
            MemorySegment xSeg = ((MemoryView<MemorySegment>) x).memory().base();
            MemorySegment ySeg = ((MemoryView<MemorySegment>) y).memory().base();

            long aBase = A.byteOffset();
            long xBase = x.byteOffset();
            long yBase = y.byteOffset();

            int vecLen = SPECIES.length();
            int vecBound = SPECIES.loopBound(n);

            // Parallelize aggressively for large matrices
            long work = (long) m * n;
            if (work < PARALLEL_MIN_WORK || m < GEMV_PARALLEL_MIN_ROWS) {
                // Sequential with loop unrolling for better ILP
                for (int row = 0; row < m; row++) {
                    long rowBase = aBase + (long) row * n * Float.BYTES;

                    // Multiple accumulators for better instruction-level parallelism
                    FloatVector acc1 = FloatVector.zero(SPECIES);
                    FloatVector acc2 = FloatVector.zero(SPECIES);
                    FloatVector acc3 = FloatVector.zero(SPECIES);
                    FloatVector acc4 = FloatVector.zero(SPECIES);

                    int col = 0;

                    // Unroll by 4 for better ILP
                    int unrollBound = (n / (vecLen * 4)) * (vecLen * 4);
                    for (; col < unrollBound; col += vecLen * 4) {
                        long aOff1 = rowBase + (long) col * Float.BYTES;
                        long xOff1 = xBase + (long) col * Float.BYTES;
                        FloatVector xv1 =
                                FloatVector.fromMemorySegment(
                                        SPECIES, xSeg, xOff1, ByteOrder.nativeOrder());
                        FloatVector av1 =
                                FloatVector.fromMemorySegment(
                                        SPECIES, aSeg, aOff1, ByteOrder.nativeOrder());
                        acc1 = av1.fma(xv1, acc1);

                        long aOff2 = aOff1 + (long) vecLen * Float.BYTES;
                        long xOff2 = xOff1 + (long) vecLen * Float.BYTES;
                        FloatVector xv2 =
                                FloatVector.fromMemorySegment(
                                        SPECIES, xSeg, xOff2, ByteOrder.nativeOrder());
                        FloatVector av2 =
                                FloatVector.fromMemorySegment(
                                        SPECIES, aSeg, aOff2, ByteOrder.nativeOrder());
                        acc2 = av2.fma(xv2, acc2);

                        long aOff3 = aOff2 + (long) vecLen * Float.BYTES;
                        long xOff3 = xOff2 + (long) vecLen * Float.BYTES;
                        FloatVector xv3 =
                                FloatVector.fromMemorySegment(
                                        SPECIES, xSeg, xOff3, ByteOrder.nativeOrder());
                        FloatVector av3 =
                                FloatVector.fromMemorySegment(
                                        SPECIES, aSeg, aOff3, ByteOrder.nativeOrder());
                        acc3 = av3.fma(xv3, acc3);

                        long aOff4 = aOff3 + (long) vecLen * Float.BYTES;
                        long xOff4 = xOff3 + (long) vecLen * Float.BYTES;
                        FloatVector xv4 =
                                FloatVector.fromMemorySegment(
                                        SPECIES, xSeg, xOff4, ByteOrder.nativeOrder());
                        FloatVector av4 =
                                FloatVector.fromMemorySegment(
                                        SPECIES, aSeg, aOff4, ByteOrder.nativeOrder());
                        acc4 = av4.fma(xv4, acc4);
                    }

                    // Handle remaining vectors
                    for (; col < vecBound; col += vecLen) {
                        long aOff = rowBase + (long) col * Float.BYTES;
                        long xOff = xBase + (long) col * Float.BYTES;
                        FloatVector xv =
                                FloatVector.fromMemorySegment(
                                        SPECIES, xSeg, xOff, ByteOrder.nativeOrder());
                        FloatVector av =
                                FloatVector.fromMemorySegment(
                                        SPECIES, aSeg, aOff, ByteOrder.nativeOrder());
                        acc1 = av.fma(xv, acc1);
                    }

                    // Combine accumulators and reduce
                    float dot = acc1.add(acc2).add(acc3).add(acc4).reduceLanes(VectorOperators.ADD);

                    // Scalar tail
                    for (; col < n; col++) {
                        float av = access.readFloat(A.memory(), rowBase + (long) col * Float.BYTES);
                        float xv = access.readFloat(x.memory(), xBase + (long) col * Float.BYTES);
                        dot += av * xv;
                    }

                    access.writeFloat(y.memory(), yBase + (long) row * Float.BYTES, dot);
                }
                return;
            }

            // Parallel processing with optimized kernel
            parallelForRows(
                    m,
                    GEMV_THREADS,
                    row -> {
                        long rowBase = aBase + (long) row * n * Float.BYTES;

                        FloatVector acc1 = FloatVector.zero(SPECIES);
                        FloatVector acc2 = FloatVector.zero(SPECIES);
                        FloatVector acc3 = FloatVector.zero(SPECIES);
                        FloatVector acc4 = FloatVector.zero(SPECIES);

                        int col = 0;
                        int unrollBound = (n / (vecLen * 4)) * (vecLen * 4);

                        for (; col < unrollBound; col += vecLen * 4) {
                            long aOff1 = rowBase + (long) col * Float.BYTES;
                            long xOff1 = xBase + (long) col * Float.BYTES;
                            FloatVector xv1 =
                                    FloatVector.fromMemorySegment(
                                            SPECIES, xSeg, xOff1, ByteOrder.nativeOrder());
                            FloatVector av1 =
                                    FloatVector.fromMemorySegment(
                                            SPECIES, aSeg, aOff1, ByteOrder.nativeOrder());
                            acc1 = av1.fma(xv1, acc1);

                            long aOff2 = aOff1 + (long) vecLen * Float.BYTES;
                            long xOff2 = xOff1 + (long) vecLen * Float.BYTES;
                            FloatVector xv2 =
                                    FloatVector.fromMemorySegment(
                                            SPECIES, xSeg, xOff2, ByteOrder.nativeOrder());
                            FloatVector av2 =
                                    FloatVector.fromMemorySegment(
                                            SPECIES, aSeg, aOff2, ByteOrder.nativeOrder());
                            acc2 = av2.fma(xv2, acc2);

                            long aOff3 = aOff2 + (long) vecLen * Float.BYTES;
                            long xOff3 = xOff2 + (long) vecLen * Float.BYTES;
                            FloatVector xv3 =
                                    FloatVector.fromMemorySegment(
                                            SPECIES, xSeg, xOff3, ByteOrder.nativeOrder());
                            FloatVector av3 =
                                    FloatVector.fromMemorySegment(
                                            SPECIES, aSeg, aOff3, ByteOrder.nativeOrder());
                            acc3 = av3.fma(xv3, acc3);

                            long aOff4 = aOff3 + (long) vecLen * Float.BYTES;
                            long xOff4 = xOff3 + (long) vecLen * Float.BYTES;
                            FloatVector xv4 =
                                    FloatVector.fromMemorySegment(
                                            SPECIES, xSeg, xOff4, ByteOrder.nativeOrder());
                            FloatVector av4 =
                                    FloatVector.fromMemorySegment(
                                            SPECIES, aSeg, aOff4, ByteOrder.nativeOrder());
                            acc4 = av4.fma(xv4, acc4);
                        }

                        for (; col < vecBound; col += vecLen) {
                            long aOff = rowBase + (long) col * Float.BYTES;
                            long xOff = xBase + (long) col * Float.BYTES;
                            FloatVector xv =
                                    FloatVector.fromMemorySegment(
                                            SPECIES, xSeg, xOff, ByteOrder.nativeOrder());
                            FloatVector av =
                                    FloatVector.fromMemorySegment(
                                            SPECIES, aSeg, aOff, ByteOrder.nativeOrder());
                            acc1 = av.fma(xv, acc1);
                        }

                        float dot =
                                acc1.add(acc2).add(acc3).add(acc4).reduceLanes(VectorOperators.ADD);

                        for (; col < n; col++) {
                            float av =
                                    access.readFloat(
                                            A.memory(), rowBase + (long) col * Float.BYTES);
                            float xv =
                                    access.readFloat(x.memory(), xBase + (long) col * Float.BYTES);
                            dot += av * xv;
                        }

                        access.writeFloat(y.memory(), yBase + (long) row * Float.BYTES, dot);
                    });
        }

        private static void parallelForRows(int rows, java.util.function.IntConsumer body) {
            parallelForRows(rows, THREADS, body);
        }

        private static void parallelForRows(
                int rows, int maxWorkers, java.util.function.IntConsumer body) {
            if (rows <= 1) {
                for (int row = 0; row < rows; row++) {
                    body.accept(row);
                }
                return;
            }
            if (USE_PERSISTENT_PARALLEL) {
                PARALLEL_POOL.forRange(rows, 2, body);
                return;
            }
            int workers = Math.min(Math.max(1, maxWorkers), rows);
            int chunk = (rows + workers - 1) / workers;
            java.util.List<java.util.concurrent.Future<?>> futures =
                    new java.util.ArrayList<>(workers);
            for (int w = 0; w < workers; w++) {
                int start = w * chunk;
                int end = Math.min(rows, start + chunk);
                if (start >= end) continue;
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

        private Tensor attention(LlamaState state, Tensor q, int layer, DecodeScratch scratch) {
            return attention(state, q, layer, scratch, state.position);
        }

        private Tensor attention(
                LlamaState state, Tensor q, int layer, DecodeScratch scratch, int position) {
            if (q.dataType() != DataType.FP32) {
                throw new IllegalStateException("attention requires FP32 tensors");
            }
            int length = position + 1;
            MemoryView<?> qView = q.materialize();
            MemoryView<?> key = state.keyCache[layer];
            MemoryView<?> value = state.valueCache[layer];
            MemoryView<?> out = scratch.attentionOutBuf;

            @SuppressWarnings({"rawtypes", "unchecked"})
            MemoryAccess access =
                    Environment.runtimeFor(qView.memory().device()).memoryDomain().directAccess();
            MemorySegment keySeg = ((MemoryView<MemorySegment>) key).memory().base();
            MemorySegment valueSeg = ((MemoryView<MemorySegment>) value).memory().base();

            int kvMul = cfg.nHeads() / cfg.nKvHeads();
            int headDim = cfg.headDim();
            float invScale = 1f / (float) Math.sqrt(headDim);
            long qBase = qView.byteOffset();
            long keyBase = key.byteOffset();
            long valueBase = value.byteOffset();
            long outBase = out.byteOffset();
            long keyRowStrideBytes = (long) (cfg.nKvHeads() * headDim) * Float.BYTES;
            long valueRowStrideBytes = (long) cfg.contextLength() * Float.BYTES;
            int vecLen = SPECIES.length();
            int vecBoundHead = SPECIES.loopBound(headDim);

            int nHeads = cfg.nHeads();
            java.util.function.IntConsumer headKernel =
                    head -> {
                        int kvHead = head / kvMul;
                        int kvHeadOffset = kvHead * headDim;
                        int headOffset = head * headDim;
                        long qHeadBase = qBase + (long) headOffset * Float.BYTES;

                        AttentionScratch local = ATTN_SCRATCH.get();
                        local.ensureCapacity(headDim, length, headDim);
                        float[] qHeadScratch = local.q;
                        float[] scoreScratch = local.scores;
                        float[] outHeadScratch = local.out;
                        float[] tmpHeadScratch = local.tmp;

                        for (int d = 0; d < headDim; d++) {
                            qHeadScratch[d] =
                                    access.readFloat(
                                            qView.memory(), qHeadBase + (long) d * Float.BYTES);
                        }
                        if (ATTN_DECODE_TWO_PASS) {
                            for (int t = 0; t < length; t++) {
                                long kHeadBase =
                                        keyBase
                                                + (long) t * keyRowStrideBytes
                                                + (long) kvHeadOffset * Float.BYTES;

                                FloatVector accVec = FloatVector.zero(SPECIES);
                                int d = 0;
                                for (; d < vecBoundHead; d += vecLen) {
                                    FloatVector qVec =
                                            FloatVector.fromArray(SPECIES, qHeadScratch, d);
                                    FloatVector kVec =
                                            FloatVector.fromMemorySegment(
                                                    SPECIES,
                                                    keySeg,
                                                    kHeadBase + (long) d * Float.BYTES,
                                                    ByteOrder.nativeOrder());
                                    accVec = qVec.fma(kVec, accVec);
                                }
                                float dot = accVec.reduceLanes(VectorOperators.ADD);
                                for (; d < headDim; d++) {
                                    float kv =
                                            access.readFloat(
                                                    key.memory(),
                                                    kHeadBase + (long) d * Float.BYTES);
                                    dot += qHeadScratch[d] * kv;
                                }
                                scoreScratch[t] = dot * invScale;
                            }

                            float maxScore = Float.NEGATIVE_INFINITY;
                            for (int t = 0; t < length; t++) {
                                if (scoreScratch[t] > maxScore) {
                                    maxScore = scoreScratch[t];
                                }
                            }
                            float sum = 0.0f;
                            for (int t = 0; t < length; t++) {
                                float v = (float) Math.exp(scoreScratch[t] - maxScore);
                                scoreScratch[t] = v;
                                sum += v;
                            }
                            float invSum = sum == 0.0f ? 0.0f : (1.0f / sum);
                            for (int t = 0; t < length; t++) {
                                scoreScratch[t] *= invSum;
                            }

                            int vecBoundTime = SPECIES.loopBound(length);
                            for (int d2 = 0; d2 < headDim; d2++) {
                                long vHeadBase =
                                        valueBase
                                                + (long) (kvHeadOffset + d2) * valueRowStrideBytes;

                                FloatVector accV = FloatVector.zero(SPECIES);
                                int t = 0;
                                for (; t < vecBoundTime; t += vecLen) {
                                    FloatVector scoreVec =
                                            FloatVector.fromArray(SPECIES, scoreScratch, t);
                                    FloatVector valueVec =
                                            FloatVector.fromMemorySegment(
                                                    SPECIES,
                                                    valueSeg,
                                                    vHeadBase + (long) t * Float.BYTES,
                                                    ByteOrder.nativeOrder());
                                    accV = scoreVec.fma(valueVec, accV);
                                }

                                float outVal = accV.reduceLanes(VectorOperators.ADD);
                                for (; t < length; t++) {
                                    float vv =
                                            access.readFloat(
                                                    value.memory(),
                                                    vHeadBase + (long) t * Float.BYTES);
                                    outVal += scoreScratch[t] * vv;
                                }
                                outHeadScratch[d2] = outVal;
                            }
                        } else {
                            for (int d = 0; d < headDim; d++) {
                                outHeadScratch[d] = 0.0f;
                            }

                            float runningMax = Float.NEGATIVE_INFINITY;
                            float runningNorm = 0.0f;
                            if (ATTN_DECODE_BLOCKED && length >= (ATTN_DECODE_BLOCK_SIZE << 1)) {
                                int blockSize = ATTN_DECODE_BLOCK_SIZE;
                                for (int blockStart = 0;
                                        blockStart < length;
                                        blockStart += blockSize) {
                                    int blockEnd = Math.min(length, blockStart + blockSize);
                                    int blockLen = blockEnd - blockStart;

                                    float blockMax = Float.NEGATIVE_INFINITY;
                                    for (int t = blockStart; t < blockEnd; t++) {
                                        long kHeadBase =
                                                keyBase
                                                        + (long) t * keyRowStrideBytes
                                                        + (long) kvHeadOffset * Float.BYTES;

                                        FloatVector accVec = FloatVector.zero(SPECIES);
                                        int d = 0;
                                        for (; d < vecBoundHead; d += vecLen) {
                                            FloatVector qVec =
                                                    FloatVector.fromArray(SPECIES, qHeadScratch, d);
                                            FloatVector kVec =
                                                    FloatVector.fromMemorySegment(
                                                            SPECIES,
                                                            keySeg,
                                                            kHeadBase + (long) d * Float.BYTES,
                                                            ByteOrder.nativeOrder());
                                            accVec = qVec.fma(kVec, accVec);
                                        }
                                        float dot = accVec.reduceLanes(VectorOperators.ADD);
                                        for (; d < headDim; d++) {
                                            float kv =
                                                    access.readFloat(
                                                            key.memory(),
                                                            kHeadBase + (long) d * Float.BYTES);
                                            dot += qHeadScratch[d] * kv;
                                        }
                                        float score = dot * invScale;
                                        scoreScratch[t] = score;
                                        if (score > blockMax) {
                                            blockMax = score;
                                        }
                                    }

                                    float newMax = Math.max(runningMax, blockMax);
                                    float prevTerm =
                                            runningNorm == 0.0f
                                                    ? 0.0f
                                                    : runningNorm
                                                            * (float) Math.exp(runningMax - newMax);

                                    float blockSum = 0.0f;
                                    for (int t = blockStart; t < blockEnd; t++) {
                                        float wExp = (float) Math.exp(scoreScratch[t] - newMax);
                                        scoreScratch[t] = wExp;
                                        blockSum += wExp;
                                    }

                                    float newNorm = prevTerm + blockSum;
                                    float coeffOld = newNorm == 0.0f ? 0.0f : prevTerm / newNorm;
                                    float invNorm = newNorm == 0.0f ? 0.0f : 1.0f / newNorm;

                                    int d = 0;
                                    for (; d < vecBoundHead; d += vecLen) {
                                        FloatVector outVec =
                                                FloatVector.fromArray(SPECIES, outHeadScratch, d);
                                        outVec = outVec.mul(coeffOld);
                                        outVec.intoArray(outHeadScratch, d);
                                    }
                                    for (; d < headDim; d++) {
                                        outHeadScratch[d] *= coeffOld;
                                    }

                                    for (int d2 = 0; d2 < headDim; d2++) {
                                        long vHeadBase =
                                                valueBase
                                                        + (long) (kvHeadOffset + d2)
                                                                * valueRowStrideBytes;
                                        FloatVector accV = FloatVector.zero(SPECIES);
                                        int bi = 0;
                                        int vecBoundBlock = SPECIES.loopBound(blockLen);
                                        for (; bi < vecBoundBlock; bi += vecLen) {
                                            int t = blockStart + bi;
                                            FloatVector scoreVec =
                                                    FloatVector.fromArray(SPECIES, scoreScratch, t);
                                            FloatVector valueVec =
                                                    FloatVector.fromMemorySegment(
                                                            SPECIES,
                                                            valueSeg,
                                                            vHeadBase + (long) t * Float.BYTES,
                                                            ByteOrder.nativeOrder());
                                            accV = scoreVec.fma(valueVec, accV);
                                        }
                                        float blockAcc = accV.reduceLanes(VectorOperators.ADD);
                                        for (; bi < blockLen; bi++) {
                                            int t = blockStart + bi;
                                            float vv =
                                                    access.readFloat(
                                                            value.memory(),
                                                            vHeadBase + (long) t * Float.BYTES);
                                            blockAcc += scoreScratch[t] * vv;
                                        }
                                        tmpHeadScratch[d2] = blockAcc * invNorm;
                                    }

                                    d = 0;
                                    for (; d < vecBoundHead; d += vecLen) {
                                        FloatVector outVec =
                                                FloatVector.fromArray(SPECIES, outHeadScratch, d);
                                        FloatVector tmpVec =
                                                FloatVector.fromArray(SPECIES, tmpHeadScratch, d);
                                        outVec = outVec.add(tmpVec);
                                        outVec.intoArray(outHeadScratch, d);
                                    }
                                    for (; d < headDim; d++) {
                                        outHeadScratch[d] += tmpHeadScratch[d];
                                    }

                                    runningMax = newMax;
                                    runningNorm = newNorm;
                                }
                            } else {
                                for (int t = 0; t < length; t++) {
                                    long kHeadBase =
                                            keyBase
                                                    + (long) t * keyRowStrideBytes
                                                    + (long) kvHeadOffset * Float.BYTES;

                                    FloatVector accVec = FloatVector.zero(SPECIES);
                                    int d = 0;
                                    for (; d < vecBoundHead; d += vecLen) {
                                        FloatVector qVec =
                                                FloatVector.fromArray(SPECIES, qHeadScratch, d);
                                        FloatVector kVec =
                                                FloatVector.fromMemorySegment(
                                                        SPECIES,
                                                        keySeg,
                                                        kHeadBase + (long) d * Float.BYTES,
                                                        ByteOrder.nativeOrder());
                                        accVec = qVec.fma(kVec, accVec);
                                    }
                                    float dot = accVec.reduceLanes(VectorOperators.ADD);
                                    for (; d < headDim; d++) {
                                        float kv =
                                                access.readFloat(
                                                        key.memory(),
                                                        kHeadBase + (long) d * Float.BYTES);
                                        dot += qHeadScratch[d] * kv;
                                    }

                                    float score = dot * invScale;
                                    float newMax = Math.max(runningMax, score);
                                    float oldScale =
                                            runningNorm == 0.0f
                                                    ? 0.0f
                                                    : (float) Math.exp(runningMax - newMax);
                                    float newWeight = (float) Math.exp(score - newMax);
                                    float newNorm = runningNorm * oldScale + newWeight;
                                    float coeffOld =
                                            newNorm == 0.0f
                                                    ? 0.0f
                                                    : (runningNorm * oldScale) / newNorm;
                                    float coeffNew = newNorm == 0.0f ? 0.0f : newWeight / newNorm;

                                    for (int d2 = 0; d2 < headDim; d2++) {
                                        float vv =
                                                access.readFloat(
                                                        value.memory(),
                                                        valueBase
                                                                + (long) (kvHeadOffset + d2)
                                                                        * valueRowStrideBytes
                                                                + (long) t * Float.BYTES);
                                        outHeadScratch[d2] =
                                                outHeadScratch[d2] * coeffOld + vv * coeffNew;
                                    }

                                    runningMax = newMax;
                                    runningNorm = newNorm;
                                }
                            }
                        }

                        long outHeadBase = outBase + (long) headOffset * Float.BYTES;
                        for (int d = 0; d < headDim; d++) {
                            access.writeFloat(
                                    out.memory(),
                                    outHeadBase + (long) d * Float.BYTES,
                                    outHeadScratch[d]);
                        }
                    };

            long attentionWork = (long) nHeads * (long) headDim * (long) length;
            if (THREADS > 1 && attentionWork >= PARALLEL_MIN_WORK) {
                parallelForBatch(nHeads, headKernel);
            } else {
                for (int head = 0; head < nHeads; head++) {
                    headKernel.accept(head);
                }
            }
            return scratch.attentionOutTensor;
        }

        private Tensor applyRoPETensor(
                Tensor input,
                int nHeads,
                int position,
                RopeTables rope,
                MemoryView<?> outBuf,
                Tensor outTensor) {
            MemoryView<?> in = input.materialize();
            @SuppressWarnings({"rawtypes", "unchecked"})
            MemoryAccess access =
                    Environment.runtimeFor(in.memory().device()).memoryDomain().directAccess();
            int headDim = cfg.headDim();
            long inBase = in.byteOffset();
            long outBase = outBuf.byteOffset();
            float[] cos = rope.cos[position];
            float[] sin = rope.sin[position];
            int half = headDim / 2;
            for (int head = 0; head < nHeads; head++) {
                int base = head * headDim;
                for (int i = 0; i < half; i++) {
                    int even = base + (i << 1);
                    int odd = even + 1;
                    float x0 = access.readFloat(in.memory(), inBase + (long) even * Float.BYTES);
                    float x1 = access.readFloat(in.memory(), inBase + (long) odd * Float.BYTES);
                    float c = cos[i];
                    float s = sin[i];
                    access.writeFloat(
                            outBuf.memory(), outBase + (long) even * Float.BYTES, x0 * c - x1 * s);
                    access.writeFloat(
                            outBuf.memory(), outBase + (long) odd * Float.BYTES, x0 * s + x1 * c);
                }
            }
            return outTensor;
        }

        private void writeKeyValueCache(
                LlamaState state, int layer, int kvDim, Tensor k, Tensor v) {
            writeKeyValueCache(state, layer, kvDim, k, v, state.position);
        }

        private void writeKeyValueCache(
                LlamaState state, int layer, int kvDim, Tensor k, Tensor v, int position) {
            MemoryView<?> kv = k.materialize();
            MemoryView<?> vv = v.materialize();
            MemoryView<?> keyCacheView = state.keyCache[layer];
            MemoryView<?> valueCacheView = state.valueCache[layer];
            @SuppressWarnings({"rawtypes", "unchecked"})
            MemoryAccess access =
                    Environment.runtimeFor(kv.memory().device()).memoryDomain().directAccess();
            long kBase = kv.byteOffset();
            long vBase = vv.byteOffset();
            long keyCacheBase = keyCacheView.byteOffset();
            long valueCacheBase = valueCacheView.byteOffset();
            long keyRowOffset = (long) position * kvDim * Float.BYTES;
            for (int i = 0; i < kvDim; i++) {
                float kVal = access.readFloat(kv.memory(), kBase + (long) i * Float.BYTES);
                float vVal = access.readFloat(vv.memory(), vBase + (long) i * Float.BYTES);
                access.writeFloat(
                        keyCacheView.memory(),
                        keyCacheBase + keyRowOffset + (long) i * Float.BYTES,
                        kVal);
                access.writeFloat(
                        valueCacheView.memory(),
                        valueCacheBase
                                + (long) i * cfg.contextLength() * Float.BYTES
                                + (long) position * Float.BYTES,
                        vVal);
            }
        }

        private void addInto(MemoryView<?> a, MemoryView<?> b, MemoryView<?> out, int n) {
            @SuppressWarnings({"rawtypes", "unchecked"})
            MemoryAccess access =
                    Environment.runtimeFor(a.memory().device()).memoryDomain().directAccess();
            long aBase = a.byteOffset();
            long bBase = b.byteOffset();
            long outBase = out.byteOffset();
            for (int i = 0; i < n; i++) {
                float av = access.readFloat(a.memory(), aBase + (long) i * Float.BYTES);
                float bv = access.readFloat(b.memory(), bBase + (long) i * Float.BYTES);
                access.writeFloat(out.memory(), outBase + (long) i * Float.BYTES, av + bv);
            }
        }
    }

    private static final class Sampler {
        private final int vocabSize;
        private final float temperature;
        private final float topP;
        private final int seed;
        private final java.util.Random random;

        Sampler(int vocabSize, float temperature, float topP, int seed) {
            this.vocabSize = vocabSize;
            this.temperature = temperature;
            this.topP = topP;
            this.seed = seed;
            this.random = new java.util.Random(seed);
        }

        int sample(MemoryView<?> logits) {
            @SuppressWarnings({"rawtypes", "unchecked"})
            MemoryAccess access =
                    Environment.runtimeFor(logits.memory().device()).memoryDomain().directAccess();
            long base = logits.byteOffset();

            float[] probs = new float[vocabSize];
            double sum = 0;
            for (int i = 0; i < vocabSize; i++) {
                float logit = access.readFloat(logits.memory(), base + (long) i * Float.BYTES);
                float prob = (float) Math.exp(logit / temperature);
                probs[i] = prob;
                sum += prob;
            }

            float invSum = (float) (1.0 / sum);
            for (int i = 0; i < vocabSize; i++) {
                probs[i] *= invSum;
            }

            float r = random.nextFloat();
            float cumsum = 0;
            for (int i = 0; i < vocabSize; i++) {
                cumsum += probs[i];
                if (r <= cumsum) return i;
            }
            return vocabSize - 1;
        }
    }

    private static final class Options {
        final Path modelPath;
        final String prompt;
        final String systemPrompt;
        final int maxTokens;
        final float temperature;
        final float topP;
        final int seed;
        final boolean interactive;
        final boolean stream;
        final boolean benchmark;
        final int benchPp;
        final int benchTg;
        final int benchmarkRuns;
        final int benchmarkWarmup;
        final int benchmarkSeed;

        Options(
                Path modelPath,
                String prompt,
                String systemPrompt,
                int maxTokens,
                float temperature,
                float topP,
                int seed,
                boolean interactive,
                boolean stream,
                boolean benchmark,
                int benchPp,
                int benchTg,
                int benchmarkRuns,
                int benchmarkWarmup,
                int benchmarkSeed) {
            this.modelPath = modelPath;
            this.prompt = prompt;
            this.systemPrompt = systemPrompt;
            this.maxTokens = maxTokens;
            this.temperature = temperature;
            this.topP = topP;
            this.seed = seed;
            this.interactive = interactive;
            this.stream = stream;
            this.benchmark = benchmark;
            this.benchPp = benchPp;
            this.benchTg = benchTg;
            this.benchmarkRuns = benchmarkRuns;
            this.benchmarkWarmup = benchmarkWarmup;
            this.benchmarkSeed = benchmarkSeed;
        }

        static Options parse(String[] args) {
            Path modelPath = null;
            String prompt = null;
            String systemPrompt = null;
            int maxTokens = 256;
            float temperature = 0.8f;
            float topP = 0.9f;
            int seed = 42;
            boolean interactive = false;
            boolean stream = false;
            boolean benchmark = false;
            int benchPp = 512;
            int benchTg = 128;
            int benchmarkRuns = 5;
            int benchmarkWarmup = 1;
            int benchmarkSeed = 1234;

            for (int i = 0; i < args.length; i++) {
                String arg = args[i];
                switch (arg) {
                    case "--model", "-m" -> modelPath = Path.of(args[++i]);
                    case "--prompt", "-p" -> prompt = args[++i];
                    case "--system-prompt" -> systemPrompt = args[++i];
                    case "--max-tokens", "-n" -> maxTokens = Integer.parseInt(args[++i]);
                    case "--temperature", "-t" -> temperature = Float.parseFloat(args[++i]);
                    case "--top-p" -> topP = Float.parseFloat(args[++i]);
                    case "--seed", "-s" -> seed = Integer.parseInt(args[++i]);
                    case "--interactive", "-i" -> interactive = true;
                    case "--stream" -> stream = true;
                    case "--bench" -> benchmark = true;
                    case "--pp" -> benchPp = Integer.parseInt(args[++i]);
                    case "--tg" -> benchTg = Integer.parseInt(args[++i]);
                    case "--bench-runs" -> benchmarkRuns = Integer.parseInt(args[++i]);
                    case "--bench-warmup" -> benchmarkWarmup = Integer.parseInt(args[++i]);
                    case "--bench-seed" -> benchmarkSeed = Integer.parseInt(args[++i]);
                    default -> throw new IllegalArgumentException("Unknown option: " + arg);
                }
            }

            if (modelPath == null) {
                throw new IllegalArgumentException("Missing required option: --model");
            }

            return new Options(
                    modelPath,
                    prompt,
                    systemPrompt,
                    maxTokens,
                    temperature,
                    topP,
                    seed,
                    interactive,
                    stream,
                    benchmark,
                    benchPp,
                    benchTg,
                    benchmarkRuns,
                    benchmarkWarmup,
                    benchmarkSeed);
        }
    }

    private static final class LoadedModel implements AutoCloseable {
        final LlamaConfig config;
        final Tokenizer tokenizer;
        final Llama3ChatFormat chatFormat;
        final LlamaModel model;
        final Arena arena;
        final FileChannel channel;

        private LoadedModel(
                LlamaConfig config,
                Tokenizer tokenizer,
                Llama3ChatFormat chatFormat,
                LlamaModel model,
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
                    config, tokenizer, format, new LlamaModel(config, weights), arena, channel);
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
        return new LlamaConfig(
                dim,
                gguf.getValue(int.class, a + ".feed_forward_length"),
                gguf.getValue(int.class, a + ".block_count"),
                heads,
                kvHeads,
                headDim,
                gguf.getValue(int.class, a + ".context_length"),
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

        LlamaLayer[] layers = new LlamaLayer[cfg.nLayers()];
        for (int i = 0; i < cfg.nLayers(); i++) {
            layers[i] =
                    new LlamaLayer(
                            mapTensorAsArrayTensor(
                                    gguf,
                                    "blk." + i + ".attn_norm.weight",
                                    tensorBase,
                                    channel,
                                    arena),
                            mapTensorAsArrayTensor(
                                    gguf,
                                    "blk." + i + ".attn_q.weight",
                                    tensorBase,
                                    channel,
                                    arena),
                            mapTensorAsArrayTensor(
                                    gguf,
                                    "blk." + i + ".attn_k.weight",
                                    tensorBase,
                                    channel,
                                    arena),
                            mapTensorAsArrayTensor(
                                    gguf,
                                    "blk." + i + ".attn_v.weight",
                                    tensorBase,
                                    channel,
                                    arena),
                            mapTensorAsArrayTensor(
                                    gguf,
                                    "blk." + i + ".attn_output.weight",
                                    tensorBase,
                                    channel,
                                    arena),
                            mapTensorAsArrayTensor(
                                    gguf,
                                    "blk." + i + ".ffn_norm.weight",
                                    tensorBase,
                                    channel,
                                    arena),
                            mapTensorAsArrayTensor(
                                    gguf,
                                    "blk." + i + ".ffn_gate.weight",
                                    tensorBase,
                                    channel,
                                    arena),
                            mapTensorAsArrayTensor(
                                    gguf,
                                    "blk." + i + ".ffn_down.weight",
                                    tensorBase,
                                    channel,
                                    arena),
                            mapTensorAsArrayTensor(
                                    gguf,
                                    "blk." + i + ".ffn_up.weight",
                                    tensorBase,
                                    channel,
                                    arena));
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
        return new LlamaWeights(tokenTable, output, outNorm, layers, rope);
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
            MemoryAccess<MemorySegment> access = Environment.nativeMemoryDomain().directAccess();
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
        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment> host = (MemoryView<MemorySegment>) view;
        @SuppressWarnings("unchecked")
        MemoryAccess<MemorySegment> access = Environment.nativeMemoryDomain().directAccess();
        int size = Math.toIntExact(host.shape().size());
        float[] out = new float[size];
        for (int i = 0; i < size; i++) {
            long off = Indexing.linearToOffset(host, i);
            out[i] = access.readFloat(host.memory(), off);
        }
        return out;
    }
}
