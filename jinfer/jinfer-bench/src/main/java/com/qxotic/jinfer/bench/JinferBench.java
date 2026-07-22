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
 * pp/tg throughput benchmark for the new com.qxotic.jinfer.models seam (the jinfer-gemma4 port),
 * printed in the same markdown table as {@link LegacyBench} so the new API's numbers are directly
 * comparable to the production engine. Drives the forward directly — {@code newState → ingest →
 * logits} — and times it with {@code nanoTime} (the seam has no internal timers). Greedy argmax
 * (temp 0), like llama-bench.
 *
 * <pre>jinfer-bench -m model.gguf [-p 512] [-n 128] [-r 5] [-w 2] [--ctx N]</pre>
 */
public final class JinferBench {

    public static void main(String[] args) throws Exception {
        List<String> models = new ArrayList<>();
        int p = 512, n = 128, reps = 5, warmup = 2, ctx = 0;
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "-m", "--model" -> models.add(args[++i]);
                case "-p", "--n-prompt" -> p = Integer.parseInt(args[++i]);
                case "-n", "--n-gen" -> n = Integer.parseInt(args[++i]);
                case "-r", "--repetitions" -> reps = Integer.parseInt(args[++i]);
                case "-w", "--warmup" -> warmup = Integer.parseInt(args[++i]);
                case "--ctx" -> ctx = Integer.parseInt(args[++i]);
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
        int threads = ForkJoinPool.commonPool().getParallelism();

        List<Row> rows = new ArrayList<>();
        for (String path : models) {
            System.err.printf("loading %s (ctx=%d) via com.qxotic.jinfer.models ...%n", path, ctx);
            LoadedModel<?> model = loadAny(Path.of(path), ctx);
            String name = name(path);
            if (p > 0) rows.add(measure(model, name, threads, "pp" + p, p, true, warmup, reps));
            if (n > 0) rows.add(measure(model, name, threads, "tg" + n, n, false, warmup, reps));
        }
        printTable(rows);
    }

    /** Arch dispatch via the shared ModelProvider services. */
    private static LoadedModel<?> loadAny(Path path, int ctx) throws Exception {
        return com.qxotic.jinfer.chat.Models.load(path, ctx);
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
        S s = model.model().newState(ctx, Math.max(prompt.length, 16));
        if (prefill) {
            // pp: one batched prefill of `count` tokens
            long t0 = System.nanoTime();
            model.model().ingest(s, Batch.prefill(prompt));
            return count / ((System.nanoTime() - t0) / 1e9);
        }
        // tg: prime with one token, then time `count` single-token decode steps
        model.model().ingest(s, Batch.prefill(prompt));
        int tok = argmax(model.model().logits(s), vocab);
        long t0 = System.nanoTime();
        for (int g = 0; g < count; g++) {
            model.model().ingest(s, Batch.step(tok));
            tok = argmax(model.model().logits(s), vocab);
        }
        return count / ((System.nanoTime() - t0) / 1e9);
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

    private static int argmax(FloatTensor t, int n) {
        int best = 0;
        for (int i = 1; i < n; i++) if (t.getFloat(i) > t.getFloat(best)) best = i;
        return best;
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
        double s = 0;
        for (double v : a) s += v;
        return s / a.length;
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
