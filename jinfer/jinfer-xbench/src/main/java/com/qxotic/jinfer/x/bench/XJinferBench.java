package com.qxotic.jinfer.x.bench;

import com.qxotic.jinfer.x.Views;
import java.io.PrintStream;
import java.lang.foreign.Arena;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ForkJoinPool;

/**
 * A/B twin of {@code com.qxotic.jinfer.bench.JinferBench}: a llama-bench-parity harness (this
 * repo's {@code tools/llama-bench/llama-bench.cpp}) driving both LFM2 and Gemma 4 implementations
 * behind {@code --impl old,x} so old vs x is one timing loop in one JVM. The llama-bench approach,
 * point by point:
 *
 * <ul>
 *   <li>pp512 / tg128 are SEPARATE tests, each with its own context sized exactly to the work
 *       ({@code n_ctx = n_prompt} for pp, {@code n_ctx = n_gen} for tg - llama-bench derives {@code
 *       n_prompt + n_gen + n_depth} per instance, and each instance sets the other count to 0);
 *       {@code --ctx} overrides both
 *   <li>ONE state per test, {@code reset()} before every warmup pass and every timed rep -
 *       llama-bench's {@code llama_memory_clear} per rep on a reused context. No allocation or
 *       page-fault cost inside the timed region
 *   <li>pp: the full prompt ingested in chunks of 512 - llama-bench's compute-graph width ({@code
 *       -ub 512}; its {@code -b 2048} only caps tokens per decode CALL, ggml still computes 512-row
 *       ubatches). One vocab projection for the last token, charged inside the timed region, as
 *       llama_decode does
 *   <li>tg: from a CLEARED (empty) state, {@code n} single-token decodes at positions 0..n-1, all
 *       timed - the first token goes in INSIDE the timed loop, like llama-bench's {@code test_gen}.
 *       Logits projected every step, never argmaxed: llama-bench feeds back {@code rand() %
 *       n_vocab}, this feeds a cheap LCG successor (throughput is content-independent for dense
 *       models; MoE routing sees pseudo-random tokens either way)
 *   <li>threads: physical cores by default ({@code common_cpu_get_num_math}), the SAME count for pp
 *       and tg ({@code llama_set_n_threads(ctx, t, t)}), plus the JAM thread-count guard
 *   <li>reporting: mean ± stddev of per-rep tokens/second - llama-bench's avg_ts / stdev_ts
 * </ul>
 *
 * Deliberate deltas, all JVM-necessitated or immaterial: warmup is ADAPTIVE (full-test passes until
 * throughput settles within 3% over a window of 3, min {@code -w}, max 30) instead of llama-bench's
 * single warmup run - native code needs one pass, the JIT needs several; {@code --no-warmup} skips
 * it. Tokens are synthetic in-range ids, not real BOS/rand. Threadpool priority/polling ({@code
 * poll=50}) has no Java analog. Loading is direct ({@code loadModel}) on both sides and never
 * timed.
 *
 * <pre>jinfer-xbench -m model.gguf [--impl old,x] [-p 512] [-n 128] [-r 5] [-w 2] [--ctx N]</pre>
 */
public final class XJinferBench {

    /**
     * Compute-graph width for prefill chunks - llama-bench's {@code -ub 512}. Pinned here rather
     * than inherited from {@code RuntimeFlags.BATCH_CAPACITY} so an ambient {@code
     * -Djinfer.batchCapacity} cannot silently change what both sides measure.
     */
    private static final int UBATCH = 512;

    public static void main(String[] args) throws Exception {
        List<String> models = new ArrayList<>();
        List<String> impls = new ArrayList<>();
        int p = 512, n = 128, reps = 5, warmup = 2, ctx = 0, threads = 0;
        boolean noWarmup = false;
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "-m", "--model" -> models.add(args[++i]);
                case "--impl" -> impls.addAll(List.of(args[++i].split(",")));
                case "-p", "--n-prompt" -> p = Integer.parseInt(args[++i]);
                case "-n", "--n-gen" -> n = Integer.parseInt(args[++i]);
                case "-r", "--repetitions" -> reps = Integer.parseInt(args[++i]);
                case "-w", "--warmup" -> warmup = Integer.parseInt(args[++i]);
                case "--no-warmup" -> noWarmup = true;
                case "--ctx" -> ctx = Integer.parseInt(args[++i]);
                case "-t", "--threads" -> threads = Integer.parseInt(args[++i]);
                case "-h", "--help" -> {
                    usage(System.out);
                    return;
                }
                default -> {
                    System.err.println("unknown option: " + args[i]);
                    usage(System.err);
                    System.exit(2);
                }
            }
        }
        if (models.isEmpty()) {
            usage(System.err);
            System.exit(2);
        }
        if (impls.isEmpty()) impls = List.of("old", "x");

        // THREAD PARITY, verbatim from JinferBench (both trees read the same
        // jinfer.decodeThreads property and pin the same common pool).
        if (threads <= 0) threads = physicalCores();
        System.setProperty("jinfer.decodeThreads", Integer.toString(threads));
        System.setProperty(
                "java.util.concurrent.ForkJoinPool.common.parallelism",
                Integer.toString(Math.max(1, threads - 1))); // +1 for the submitting thread
        int prefillThreads = ForkJoinPool.commonPool().getParallelism() + 1;
        int decodeThreads = com.qxotic.jinfer.RuntimeFlags.DECODE_THREADS;
        System.err.printf(
                "threads: prefill=%d decode=%d (requested %d)%n",
                prefillThreads, decodeThreads, threads);
        String jamThreads = System.getenv("JAM_NUM_THREADS");
        boolean jamAgrees =
                jamThreads == null
                        ? threads == physicalCores()
                        : jamThreads.trim().equals(Integer.toString(threads));
        if (!jamAgrees) {
            System.err.printf(
                    "WARNING: JAM_NUM_THREADS=%s, so the native gemm backend is NOT at %d threads"
                            + " and pp is not comparable to llama-bench -t %d.%n         Re-run as:"
                            + " JAM_NUM_THREADS=%d jinfer-xbench ...%n",
                    jamThreads == null ? "<unset, jam auto-picks physical cores>" : jamThreads,
                    threads,
                    threads,
                    threads);
        }

        List<Row> rows = new ArrayList<>();
        for (String path : models) {
            for (String implName : impls) {
                System.err.printf("loading %s impl=%s ...%n", path, implName);
                BenchModel impl = load(Path.of(path), implName);
                String name = name(path) + " [" + implName + "]";
                // llama-bench: pp and tg are separate tests on separate contexts sized to the
                // work (n_ctx = n_prompt resp. n_gen; --ctx overrides both).
                if (p > 0)
                    rows.add(
                            measure(
                                    impl,
                                    name,
                                    prefillThreads,
                                    "pp" + p,
                                    p,
                                    true,
                                    ctx != 0 ? ctx : p,
                                    warmup,
                                    reps,
                                    noWarmup));
                if (n > 0)
                    rows.add(
                            measure(
                                    impl,
                                    name,
                                    decodeThreads,
                                    "tg" + n,
                                    n,
                                    false,
                                    ctx != 0 ? ctx : n,
                                    warmup,
                                    reps,
                                    noWarmup));
            }
        }
        printTable(rows);
    }

    /**
     * One impl = load + the operations {@code runOnce} times, each a transliteration of what
     * llama-bench's {@code test_prompt}/{@code test_gen} do on the llama.cpp context. One state per
     * test, {@link #reset()} per pass - llama-bench's per-rep {@code llama_memory_clear}.
     */
    private abstract static class BenchModel {
        abstract int vocab();

        abstract void newState(int ctx);

        /** llama_memory_clear: KV/conv state back to empty, same buffers. */
        abstract void reset();

        abstract void ingestPrefill(int[] prompt);

        abstract void ingestStep(int tok);

        abstract float logits0();
    }

    /** Old tree: exactly llama-bench's calls transliterated to a directly-loaded old Lfm2. */
    private static final class OldBenchModel extends BenchModel {
        private final com.qxotic.jinfer.models.lfm2.Lfm2 model;
        private com.qxotic.jinfer.models.lfm2.Lfm2.State s;

        OldBenchModel(Path path) throws Exception {
            this.model = com.qxotic.jinfer.models.lfm2.Lfm2.loadModel(path, Arena.ofAuto());
        }

        @Override
        int vocab() {
            return model.config().vocabularySize();
        }

        @Override
        void newState(int ctx) {
            s = model.newState(ctx, UBATCH); // owned arena, GC/Cleaner-freed
        }

        @Override
        void reset() {
            s.reset();
        }

        @Override
        void ingestPrefill(int[] prompt) {
            List<com.qxotic.jinfer.Batch> chunks =
                    com.qxotic.jinfer.Batch.prepare(
                            List.of(com.qxotic.jinfer.Batch.prefill(prompt)), s.batchCapacity());
            for (com.qxotic.jinfer.Batch b : chunks) model.ingest(s, b);
        }

        @Override
        void ingestStep(int tok) {
            model.ingest(s, com.qxotic.jinfer.Batch.step(tok));
        }

        @Override
        float logits0() {
            return model.logits(s).getFloat(0);
        }
    }

    /** X tree: the same calls transliterated to the x seam (manual chunking, per-call claim). */
    private static final class XBenchModel extends BenchModel {
        private final com.qxotic.jinfer.x.models.lfm2.Lfm2 model;
        private com.qxotic.jinfer.x.models.lfm2.Lfm2.State s;

        XBenchModel(Path path) throws Exception {
            this.model = com.qxotic.jinfer.x.models.lfm2.Lfm2.loadModel(path, Arena.ofAuto());
        }

        @Override
        int vocab() {
            return model.config().vocabularySize();
        }

        @Override
        void newState(int ctx) {
            s = model.newState(ctx, UBATCH); // owned arena, GC/Cleaner-freed
        }

        @Override
        void reset() {
            s.reset();
        }

        @Override
        void ingestPrefill(int[] prompt) {
            for (com.qxotic.jinfer.x.boundary.Batch b :
                    com.qxotic.jinfer.x.boundary.Batch.prepare(
                            List.of(com.qxotic.jinfer.x.boundary.Batch.prefill(prompt)),
                            s.batchCapacity())) {
                model.ingest(s, b);
            }
        }

        @Override
        void ingestStep(int tok) {
            model.ingest(s, com.qxotic.jinfer.x.boundary.Batch.step(tok));
        }

        @Override
        float logits0() {
            return Views.getFloat(
                    Views.castToSegmentBacked(model.logits(s), "logits"), 0, "logits");
        }
    }

    private static final class OldGemma4BenchModel extends BenchModel {
        private final com.qxotic.jinfer.models.gemma4.Gemma4 model;
        private com.qxotic.jinfer.models.gemma4.Gemma4.State s;

        OldGemma4BenchModel(Path path) throws Exception {
            this.model = com.qxotic.jinfer.models.gemma4.Gemma4.loadModel(path, Arena.ofAuto());
        }

        @Override
        int vocab() {
            return model.config().vocabularySize();
        }

        @Override
        void newState(int ctx) {
            s = model.newState(ctx, UBATCH);
        }

        @Override
        void reset() {
            s.reset();
        }

        @Override
        void ingestPrefill(int[] prompt) {
            for (com.qxotic.jinfer.Batch b :
                    com.qxotic.jinfer.Batch.prepare(
                            List.of(com.qxotic.jinfer.Batch.prefill(prompt)), s.batchCapacity())) {
                model.ingest(s, b);
            }
        }

        @Override
        void ingestStep(int tok) {
            model.ingest(s, com.qxotic.jinfer.Batch.step(tok));
        }

        @Override
        float logits0() {
            return model.logits(s).getFloat(0);
        }
    }

    private static final class XGemma4BenchModel extends BenchModel {
        private final com.qxotic.jinfer.x.models.gemma4.Gemma4 model;
        private com.qxotic.jinfer.x.models.gemma4.Gemma4.State s;

        XGemma4BenchModel(Path path) throws Exception {
            this.model = com.qxotic.jinfer.x.models.gemma4.Gemma4.loadModel(path, Arena.ofAuto());
        }

        @Override
        int vocab() {
            return model.config().vocabularySize();
        }

        @Override
        void newState(int ctx) {
            s = model.newState(ctx, UBATCH);
        }

        @Override
        void reset() {
            s.reset();
        }

        @Override
        void ingestPrefill(int[] prompt) {
            for (com.qxotic.jinfer.x.boundary.Batch b :
                    com.qxotic.jinfer.x.boundary.Batch.prepare(
                            List.of(com.qxotic.jinfer.x.boundary.Batch.prefill(prompt)),
                            s.batchCapacity())) {
                model.ingest(s, b);
            }
        }

        @Override
        void ingestStep(int tok) {
            model.ingest(s, com.qxotic.jinfer.x.boundary.Batch.step(tok));
        }

        @Override
        float logits0() {
            return Views.getFloat(
                    Views.castToSegmentBacked(model.logits(s), "logits"), 0, "logits");
        }
    }

    private static BenchModel load(Path path, String impl) throws Exception {
        String architecture;
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            architecture =
                    com.qxotic.jinfer.x.kernels.ModelLoader.readGguf(channel, path.toString())
                            .getString("general.architecture");
        }
        return switch (architecture + ":" + impl) {
            case "lfm2:old" -> new OldBenchModel(path);
            case "lfm2:x" -> new XBenchModel(path);
            case "gemma4:old" -> new OldGemma4BenchModel(path);
            case "gemma4:x" -> new XGemma4BenchModel(path);
            default -> {
                System.err.println("unsupported architecture/impl: " + architecture + ":" + impl);
                System.exit(2);
                throw new AssertionError();
            }
        };
    }

    /**
     * Physical cores, the same quantity llama-bench defaults to. Duplicated from RuntimeFlags
     * rather than imported because it must run BEFORE RuntimeFlags is initialized.
     */
    private static int physicalCores() {
        int logical = Runtime.getRuntime().availableProcessors();
        try {
            boolean smt =
                    !"0"
                            .equals(
                                    Files.readString(Path.of("/sys/devices/system/cpu/smt/active"))
                                            .trim());
            return smt ? Math.max(1, logical / 2) : logical;
        } catch (Exception notLinux) {
            String arch = System.getProperty("os.arch", "");
            return arch.contains("aarch64") || arch.contains("arm")
                    ? logical
                    : Math.max(1, logical / 2);
        }
    }

    /**
     * One llama-bench test: ONE state sized to the work, adaptive warmup (llama-bench's single
     * warmup pass is not enough for the JIT), then timed reps. Every pass starts with {@code
     * reset()} - llama-bench's per-rep {@code llama_memory_clear}.
     */
    private static Row measure(
            BenchModel model,
            String name,
            int threads,
            String test,
            int count,
            boolean prefill,
            int ctx,
            int minWarmup,
            int reps,
            boolean noWarmup) {
        int vocab = model.vocab();
        int[] prompt = prefill ? fillerTokens(vocab, count) : null;
        model.newState(ctx);
        System.err.printf("  %-6s state: ctx=%d batch=%d%n", test, ctx, UBATCH);

        if (noWarmup) {
            System.err.printf("  %-6s warmup skipped (--no-warmup)%n", test);
        } else {
            final double TOL = 0.03;
            final int WINDOW = 3, MAX = Math.max(minWarmup, 30);
            double[] recent = new double[WINDOW];
            int passes = 0;
            while (passes < MAX) {
                double t = runOnce(model, prompt, count, prefill, vocab);
                recent[passes % WINDOW] = t;
                passes++;
                System.err.printf("  %-6s [warmup %2d] %8.2f t/s%n", test, passes, t);
                if (passes >= Math.max(minWarmup, WINDOW)) {
                    double lo = Double.MAX_VALUE, hi = 0;
                    for (double v : recent) {
                        lo = Math.min(lo, v);
                        hi = Math.max(hi, v);
                    }
                    if ((hi - lo) / lo < TOL) break;
                }
            }
            System.err.printf("  %-6s stabilized after %d warmup passes%n", test, passes);
        }

        double[] tps = new double[reps];
        for (int i = 0; i < reps; i++) {
            tps[i] = runOnce(model, prompt, count, prefill, vocab);
            System.err.printf("  %-6s [rep    %2d] %8.2f t/s%n", test, i, tps[i]);
        }
        return new Row(name, threads, test, mean(tps), stddev(tps));
    }

    /**
     * One pass, timed exactly as llama-bench times: the clock spans the prompt processing (pp)
     * resp. the full generation loop from an EMPTY state (tg), nothing else. Both include the vocab
     * projection llama_decode computes for the last token of a batch / each step.
     */
    private static double runOnce(
            BenchModel model, int[] prompt, int count, boolean prefill, int vocab) {
        model.reset(); // llama_memory_clear, outside the timed region
        if (prefill) {
            long t0 = System.nanoTime();
            model.ingestPrefill(prompt);
            sink += model.logits0();
            return count / ((System.nanoTime() - t0) / 1e9);
        }
        int tok = nextToken(1, vocab); // llama-bench's BOS-or-rand first token, fed inside the loop
        long t0 = System.nanoTime();
        for (int g = 0; g < count; g++) {
            model.ingestStep(tok);
            // Project to logits (llama_decode does) but do NOT argmax: llama-bench never reads
            // them, it feeds back `rand() % n_vocab`.
            sink += model.logits0();
            tok = nextToken(tok, vocab);
        }
        return count / ((System.nanoTime() - t0) / 1e9);
    }

    /** Consumed so the logits projection cannot be optimized away; never read. */
    private static volatile float sink;

    /** A cheap in-range successor, standing in for llama-bench's {@code rand() % n_vocab}. */
    private static int nextToken(int previous, int vocab) {
        return (previous * 1103515245 + 12345 & 0x7fffffff) % vocab;
    }

    /**
     * Synthetic in-range token ids — throughput is content-independent, and tokenizer() isn't on
     * the interface.
     */
    private static int[] fillerTokens(int vocab, int count) {
        int[] ids = new int[count];
        for (int i = 0; i < count; i++) ids[i] = (i * 17 + 1) % vocab;
        return ids;
    }

    private record Row(String model, int threads, String test, double mean, double stddev) {}

    private static void printTable(List<Row> rows) {
        int w = rows.stream().mapToInt(r -> r.model.length()).max().orElse(5);
        w = Math.max(w, "model".length());
        String fmt = "| %-" + w + "s | %7s | %-6s | %16s |%n";
        System.out.printf(fmt, "model", "threads", "test", "t/s");
        System.out.printf(fmt, "-".repeat(w), "------:", "------", "---------------:");
        for (Row r : rows)
            System.out.printf(
                    fmt,
                    r.model,
                    r.threads,
                    r.test,
                    String.format("%.2f ± %.2f", r.mean, r.stddev));
    }

    private static double mean(double[] a) {
        return Arrays.stream(a).average().orElse(0);
    }

    private static double stddev(double[] a) {
        if (a.length < 2) return 0;
        double m = mean(a), s = 0;
        for (double v : a) s += (v - m) * (v - m);
        return Math.sqrt(s / (a.length - 1));
    }

    private static String name(String path) {
        String f = Path.of(path).getFileName().toString();
        return f.endsWith(".gguf") ? f.substring(0, f.length() - 5) : f;
    }

    private static void usage(PrintStream out) {
        out.println(
                """
                jinfer-xbench — llama-bench-parity harness A/B-ing old vs x ports

                usage: jinfer-xbench -m <model.gguf> [-m ...] [options]
                  -m, --model <path>      LFM2 or Gemma 4 model to benchmark (repeatable)
                      --impl <old,x>      which tree(s) to run (default old,x)
                  -p, --n-prompt <N>      prefill tokens (default 512; 0 to skip pp)
                  -n, --n-gen <N>         decode tokens  (default 128; 0 to skip tg)
                  -r, --repetitions <N>   timed reps     (default 5)
                  -w, --warmup <N>        min warmup passes; warms adaptively until throughput settles (default 2)
                      --no-warmup         skip warmup runs before benchmarking
                  -t, --threads <N>       pp and tg threads (default physical cores)
                      --ctx <N>           override context size for both tests
                                          (default per test, as llama-bench: p for pp, n for tg)\
                """);
    }
}
