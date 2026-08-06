package com.qxotic.jinfer.bench;

import com.qxotic.jinfer.*;
import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.kernels.*;
import com.qxotic.jinfer.llm.*;
import java.io.PrintStream;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ForkJoinPool;

/**
 * pp/tg throughput benchmark, printed as llama-bench's markdown table so the two are directly
 * comparable. Drives the forward seam ({@code newState -> ingest -> logits}) and times it with
 * {@code nanoTime}.
 *
 * <pre>jinfer-bench -m model.gguf [-p 512] [-n 128] [-r 5] [-w 2] [--ctx N]</pre>
 *
 * <h2>Parity with llama-bench</h2>
 *
 * Matched deliberately, and verified against {@code tools/llama-bench/llama-bench.cpp}:
 *
 * <ul>
 *   <li>defaults: {@code -p 512 -n 128 -r 5}, same as llama-bench's
 *   <li>prompt chunking at the state's batch capacity (512), llama-bench's {@code n_ubatch}
 *   <li>KV cache F16 both sides ({@code type_k}/{@code type_v} default to F16 there)
 *   <li>fresh state per rep here == {@code llama_memory_clear} per rep there
 *   <li>token content is synthetic in-range ids both sides - throughput is content-independent
 *   <li>ONE vocab projection per pp batch and per tg step, because {@code llama_decode} projects
 *       the last token of every batch. jinfer's {@code ingest} stops at the hidden states, so the
 *       bench calls {@code logits} explicitly rather than measure itself doing less work
 *   <li>NO argmax in the timed loop: llama-bench never reads the logits, it feeds back {@code
 *       rand() % n_vocab}. Scanning a 262k-entry vocab per step is a tax the reference does not pay
 * </ul>
 *
 * <h2>What the CALLER must equalize</h2>
 *
 * <ul>
 *   <li><b>Threads.</b> {@code -t} defaults to PHYSICAL cores, the same quantity llama-bench
 *       defaults to, and pins the decode pool and the common pool (which otherwise sizes to LOGICAL
 *       cpus - 2x on an SMT box). It CANNOT pin the native gemm backend: that pool is sized from an
 *       environment variable and Java cannot setenv itself, so run {@code JAM_NUM_THREADS=<t>} or
 *       the bench warns and the pp number is not comparable. Measured on a 16P/32L box: pp512 1836
 *       t/s unpinned against 1669 pinned.
 *   <li><b>Flash attention.</b> llama-bench defaults to {@code -fa auto}; force it on or off on
 *       both sides if you care which path you are measuring.
 *   <li><b>Warmup.</b> llama-bench runs one warmup pass; a JVM needs more, so this warms adaptively
 *       to a stable window. That is not a thumb on the scale - it is the same steady state
 *       llama-bench reaches on its first pass.
 * </ul>
 */
public final class JinferBench {

    public static void main(String[] args) throws Exception {
        List<String> models = new ArrayList<>();
        int p = 512, n = 128, reps = 5, warmup = 2, ctx = 0, threads = 0;
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "-m", "--model" -> models.add(args[++i]);
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
        if (ctx == 0) ctx = Math.max(p, 1) + n + 64;

        // THREAD PARITY, and it is the difference between a comparison and a press release.
        // llama-bench runs BOTH tests at common_cpu_get_num_math() = physical cores. jinfer
        // decodes at physical width already, but prefills on the common pool, which sizes to
        // LOGICAL cpus - on an SMT box that is 2x llama-bench's width for pp.
        //
        // Set BEFORE any jinfer class loads: RuntimeFlags reads jinfer.decodeThreads in a static
        // initializer, and the common pool's parallelism is fixed the first time it is touched.
        if (threads <= 0) threads = physicalCores();
        System.setProperty("jinfer.decodeThreads", Integer.toString(threads));
        System.setProperty(
                "java.util.concurrent.ForkJoinPool.common.parallelism",
                Integer.toString(Math.max(1, threads - 1))); // +1 for the submitting thread
        int prefillThreads = ForkJoinPool.commonPool().getParallelism() + 1;
        int decodeThreads = RuntimeFlags.DECODE_THREADS;
        System.err.printf(
                "threads: prefill=%d decode=%d (requested %d)%n",
                prefillThreads, decodeThreads, threads);
        // The native gemm backend has its OWN pool, sized from its own topology probe, and Java
        // cannot setenv itself. Unpinned it runs at every logical cpu - measured 1836 vs 1669
        // pp512 t/s on a 16P/32L box, i.e. a tenth of the number is threads llama-bench is not
        // using. Say so loudly rather than print a 16-thread heading over a 32-thread run.
        // jam's own AUTO is one thread per physical performance core, which is exactly this
        // bench's default - so the mismatch is only real when the env var disagrees, or when -t
        // asked for something other than physical cores (jam would still auto-pick physical).
        String jamThreads = System.getenv("JAM_NUM_THREADS");
        boolean jamAgrees =
                jamThreads == null
                        ? threads == physicalCores()
                        : jamThreads.trim().equals(Integer.toString(threads));
        if (!jamAgrees) {
            System.err.printf(
                    "WARNING: JAM_NUM_THREADS=%s, so the native gemm backend is NOT at %d threads"
                            + " and pp is not comparable to llama-bench -t %d.%n         Re-run as:"
                            + " JAM_NUM_THREADS=%d jinfer-bench ...%n",
                    jamThreads == null ? "<unset, jam auto-picks physical cores>" : jamThreads,
                    threads,
                    threads,
                    threads);
        }

        List<Row> rows = new ArrayList<>();
        for (String path : models) {
            System.err.printf("loading %s (ctx=%d) via com.qxotic.jinfer.models ...%n", path, ctx);
            LoadedModel<?> model = loadAny(Path.of(path), ctx);
            String name = name(path);
            if (p > 0)
                rows.add(measure(model, name, prefillThreads, "pp" + p, p, true, warmup, reps));
            if (n > 0)
                rows.add(measure(model, name, decodeThreads, "tg" + n, n, false, warmup, reps));
        }
        printTable(rows);
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
                                    java.nio.file.Files.readString(
                                                    Path.of("/sys/devices/system/cpu/smt/active"))
                                            .trim());
            return smt ? Math.max(1, logical / 2) : logical;
        } catch (Exception notLinux) {
            String arch = System.getProperty("os.arch", "");
            return arch.contains("aarch64") || arch.contains("arm")
                    ? logical
                    : Math.max(1, logical / 2);
        }
    }

    /** Arch dispatch via the shared ModelProvider services. */
    private static LoadedModel<?> loadAny(Path path, int ctx) throws Exception {
        return com.qxotic.jinfer.chat.Models.load(path, java.lang.foreign.Arena.ofAuto());
    }

    /**
     * One pp/tg test on the new seam. Unlike the native llama-bench (which needs a single warmup
     * run), the JVM must reach JIT/GC steady state first, so this warms <em>adaptively</em>: it
     * keeps running the pass until the last {@code WINDOW} throughputs span less than {@code TOL}
     * (after at least {@code minWarmup} passes, capped so a noisy box still terminates), then runs
     * {@code reps} timed passes reported as throughput mean ± stddev.
     */
    private static <S extends RuntimeState> Row measure(
            LoadedModel<S> model,
            String name,
            int threads,
            String test,
            int count,
            boolean prefill,
            int minWarmup,
            int reps) {
        int ctx = model.model().config().contextLength();
        int vocab = model.model().config().vocabularySize();
        int[] prompt = fillerTokens(vocab, prefill ? count : 1);

        // Adaptive warmup: run until the last WINDOW passes agree within TOL (JIT/GC settled).
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

        // Timed passes: fresh state each, mirroring llama-bench's per-rep memory_clear.
        double[] tps = new double[reps];
        for (int i = 0; i < reps; i++) {
            tps[i] = runOnce(model, ctx, prompt, count, prefill, vocab);
            System.err.printf("  %-6s [rep    %2d] %8.2f t/s%n", test, i, tps[i]);
        }
        return new Row(name, threads, test, mean(tps), stddev(tps));
    }

    /**
     * One timed pass on a fresh state: a single batched prefill (pp) or {@code count} decode steps
     * (tg). Returns tokens/second. Shared by the warmup and timed loops so both measure identical
     * work.
     */
    private static <S extends RuntimeState> double runOnce(
            LoadedModel<S> model, int ctx, int[] prompt, int count, boolean prefill, int vocab) {
        S s = model.model().newState(ctx);
        // chunked exactly like the engine (and llama-bench's default n_ubatch=512): one giant
        // batch would also blow the per-batch scratch working set past cache
        List<Batch> chunks = Batch.prepare(List.of(Batch.prefill(prompt)), s.batchCapacity());
        if (prefill) {
            long t0 = System.nanoTime();
            for (Batch b : chunks) model.model().ingest(s, b);
            // llama_decode projects the LAST token of a batch to logits, so pp pays one vocab
            // projection. jinfer's ingest stops at the hidden states, so charge it explicitly or
            // pp is measured doing strictly less work than the reference.
            sink += model.model().logits(s).getFloat(0);
            return count / ((System.nanoTime() - t0) / 1e9);
        }
        // tg: prime with one token, then time `count` single-token decode steps
        for (Batch b : chunks) model.model().ingest(s, b);
        int tok = nextToken(prompt[0], vocab);
        long t0 = System.nanoTime();
        for (int g = 0; g < count; g++) {
            model.model().ingest(s, Batch.step(tok));
            // Project to logits (llama_decode does) but do NOT argmax: llama-bench never reads
            // them, it feeds back `rand() % n_vocab`. An argmax here is a vocab-wide scan per
            // step - 262k reads on Gemma - that the reference does not pay, so it would tax
            // jinfer's tg for nothing. One float keeps the projection from being dead code.
            sink += model.model().logits(s).getFloat(0);
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
        return java.util.Arrays.stream(a).average().orElse(0);
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
                jinfer-bench — pp/tg throughput for the com.qxotic.jinfer.models seam (jinfer-gemma4)

                usage: jinfer-bench -m <model.gguf> [-m ...] [options]
                  -m, --model <path>      model to benchmark (repeatable)
                  -p, --n-prompt <N>      prefill tokens (default 512; 0 to skip pp)
                  -n, --n-gen <N>         decode tokens  (default 128; 0 to skip tg)
                  -r, --repetitions <N>   timed reps     (default 5)
                  -w, --warmup <N>        min warmup passes; warms adaptively until throughput settles (default 2)
                      --ctx <N>           context size   (default p + n + 64)\
                """);
    }
}
