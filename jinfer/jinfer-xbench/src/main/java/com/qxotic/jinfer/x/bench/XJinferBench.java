package com.qxotic.jinfer.x.bench;

import com.qxotic.jinfer.x.Segments;
import com.qxotic.jinfer.x.Views;
import java.io.PrintStream;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ForkJoinPool;

/**
 * A/B twin of {@code com.qxotic.jinfer.bench.JinferBench}: the SAME llama-bench-parity harness
 * (defaults, adaptive warmup, per-rep fresh state, synthetic ids, one vocab projection per pp batch
 * / tg step, NO argmax in the timed loop, same thread pinning and JAM warning), driving both LFM2
 * implementations behind {@code --impl old,x} so old vs x is one timing loop in one JVM. The only
 * deliberate deltas from JinferBench:
 *
 * <ul>
 *   <li>loading is direct ({@code Lfm2.loadModel}) on both sides — the slice has no ServiceLoader
 *       {@code Models} front, and load time is not measured anyway
 *   <li>the model seam is abstracted ({@link BenchModel}) instead of {@code LoadedModel<S>}
 * </ul>
 *
 * Both sides drive the CLAIMING public API ({@code ingest}/{@code logits} — per-call claim,
 * reachability fence) and chunk prefill through their own {@code Batch.prepare} — identical chunk
 * sequences (equivalence is property-tested in {@code BatchTest}); the x prepare skips the old
 * one's concat copy on a legal single batch, a few hundred ns inside the timed region, immaterial
 * against ~3ms of forward.
 *
 * <pre>jinfer-xbench -m model.gguf [--impl old,x] [-p 512] [-n 128] [-r 5] [-w 2] [--ctx N]</pre>
 */
public final class XJinferBench {

    public static void main(String[] args) throws Exception {
        List<String> models = new ArrayList<>();
        List<String> impls = new ArrayList<>();
        int p = 512, n = 128, reps = 5, warmup = 2, ctx = 0, threads = 0;
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "-m", "--model" -> models.add(args[++i]);
                case "--impl" -> impls.addAll(List.of(args[++i].split(",")));
                case "-p", "--n-prompt" -> p = Integer.parseInt(args[++i]);
                case "-n", "--n-gen" -> n = Integer.parseInt(args[++i]);
                case "-r", "--repetitions" -> reps = Integer.parseInt(args[++i]);
                case "-w", "--warmup" -> warmup = Integer.parseInt(args[++i]);
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
        if (ctx == 0) ctx = Math.max(p, 1) + n + 64;

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
                System.err.printf("loading %s (ctx=%d) impl=%s ...%n", path, ctx, implName);
                BenchModel impl = load(Path.of(path), implName);
                String name = name(path) + " [" + implName + "]";
                if (p > 0)
                    rows.add(
                            measure(
                                    impl,
                                    name,
                                    prefillThreads,
                                    "pp" + p,
                                    p,
                                    true,
                                    ctx,
                                    warmup,
                                    reps));
                if (n > 0)
                    rows.add(
                            measure(
                                    impl,
                                    name,
                                    decodeThreads,
                                    "tg" + n,
                                    n,
                                    false,
                                    ctx,
                                    warmup,
                                    reps));
            }
        }
        printTable(rows);
    }

    /**
     * One impl = load + the four operations {@code runOnce} times, each a verbatim transliteration
     * of what JinferBench's {@code runOnce} calls on the old seam. State is per-rep and owned by
     * the bench loop, so the seam is stateful: {@link #newState}/{@link #ingestPrefill}/ {@link
     * #ingestStep}/{@link #logits0}.
     */
    private abstract static class BenchModel {
        abstract int vocab();

        abstract void newState(int ctx);

        abstract void ingestPrefill(int[] prompt);

        abstract void ingestStep(int tok);

        abstract float logits0();
    }

    /** Old tree: exactly JinferBench's calls on a directly-loaded old Lfm2. */
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
            s = model.newState(ctx); // owned arena, GC/Cleaner-freed — as JinferBench
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
            s = model.newState(ctx); // owned arena, GC/Cleaner-freed — as JinferBench
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
            Views.Raw r = Views.rawF32(model.logits(s), "logits");
            return Segments.readFloat(r.vseg(), r.vbase());
        }
    }

    private static BenchModel load(Path path, String impl) throws Exception {
        return switch (impl) {
            case "old" -> new OldBenchModel(path);
            case "x" -> new XBenchModel(path);
            default -> {
                System.err.println("unknown impl: " + impl + " (expected old|x)");
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

    /** Verbatim JinferBench.measure: adaptive warmup to a stable window, then timed reps. */
    private static Row measure(
            BenchModel model,
            String name,
            int threads,
            String test,
            int count,
            boolean prefill,
            int ctx,
            int minWarmup,
            int reps) {
        int vocab = model.vocab();
        int[] prompt = fillerTokens(vocab, prefill ? count : 1);

        final double TOL = 0.03;
        final int WINDOW = 3, MAX = Math.max(minWarmup, 30);
        double[] recent = new double[WINDOW];
        int passes = 0;
        while (passes < MAX) {
            double t = runOnce(model, ctx, prompt, count, prefill, vocab);
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

        double[] tps = new double[reps];
        for (int i = 0; i < reps; i++) {
            tps[i] = runOnce(model, ctx, prompt, count, prefill, vocab);
            System.err.printf("  %-6s [rep    %2d] %8.2f t/s%n", test, i, tps[i]);
        }
        return new Row(name, threads, test, mean(tps), stddev(tps));
    }

    /** Verbatim JinferBench.runOnce, on the abstracted seam. */
    private static double runOnce(
            BenchModel model, int ctx, int[] prompt, int count, boolean prefill, int vocab) {
        model.newState(ctx);
        if (prefill) {
            long t0 = System.nanoTime();
            model.ingestPrefill(prompt);
            // llama_decode projects the LAST token of a batch to logits, so pp pays one vocab
            // projection — charged explicitly, as JinferBench does.
            sink += model.logits0();
            return count / ((System.nanoTime() - t0) / 1e9);
        }
        model.ingestPrefill(prompt);
        int tok = nextToken(prompt[0], vocab);
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
                jinfer-xbench — JinferBench's harness A/B-ing old Lfm2 vs the x port

                usage: jinfer-xbench -m <model.gguf> [-m ...] [options]
                  -m, --model <path>      model to benchmark (repeatable)
                      --impl <old,x>      which tree(s) to run (default old,x)
                  -p, --n-prompt <N>      prefill tokens (default 512; 0 to skip pp)
                  -n, --n-gen <N>         decode tokens  (default 128; 0 to skip tg)
                  -r, --repetitions <N>   timed reps     (default 5)
                  -w, --warmup <N>        min warmup passes; warms adaptively until throughput settles (default 2)
                      --ctx <N>           context size   (default p + n + 64)\
                """);
    }
}
