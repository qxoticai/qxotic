package com.qxotic.jinfer;

import java.io.PrintStream;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ForkJoinPool;

/**
 * llama-bench-style benchmark for jinfer: measures prompt processing (pp, prefill) and token
 * generation (tg, decode) throughput, warmed up, and prints a markdown table — so numbers are
 * directly comparable to {@code llama-bench}.
 *
 * <pre>
 *   jinfer-bench -m model.gguf [-m model2.gguf ...] [-p 512] [-n 128] [-r 5] [-w 2] [--ctx 4096]
 * </pre>
 *
 * pp&lt;P&gt;: prefill P tokens, throughput = P / prefill-time. tg&lt;N&gt;: generate N tokens from a
 * one-token prompt (near-empty context, like llama-bench), throughput = N / decode-time.
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
                case "-h", "--help" -> { usage(System.out); return; }
                default -> { System.err.println("unknown option: " + args[i]); usage(System.err); System.exit(2); }
            }
        }
        if (models.isEmpty()) { usage(System.err); System.exit(2); }
        if (ctx == 0) ctx = Math.max(p, 1) + n + 64;       // headroom for pp + tg
        int threads = ForkJoinPool.commonPool().getParallelism();

        List<Row> rows = new ArrayList<>();
        for (String path : models) {
            System.err.printf("loading %s (ctx=%d) ...%n", path, ctx);
            Model model = ModelLoader.loadModel(Path.of(path), ctx);
            String name = name(path);
            if (p > 0) rows.add(measure(model, name, threads, "pp" + p, p, true, warmup, reps));
            if (n > 0) rows.add(measure(model, name, threads, "tg" + n, n, false, warmup, reps));
        }
        printTable(rows);
    }

    /** One pp/tg test: warmup then {@code reps} timed passes; returns throughput mean ± stddev. */
    private static Row measure(Model model, String name, int threads, String test, int count, boolean prefill,
                               int warmup, int reps) {
        Sampler sampler = Engine.configuredSampler(model, false, 0.0f, 1.0f, 42);
        List<Integer> prompt = prefill ? fillerTokens(model, count) : fillerTokens(model, 1);
        int budget = prefill ? 1 : count;                  // pp: cross the prefill/decode boundary; tg: generate count
        Engine.Params params = new Engine.Params(sampler, budget, 0, new Engine.StopSpec(Set.of(), List.of()), false);
        Engine.Listener noop = new Engine.Listener(t -> {}, null, null, null);

        double[] tps = new double[reps];
        for (int i = 0; i < warmup + reps; i++) {
            InferenceState state = model.createNewState();
            Engine.GenerationResult r = Engine.generate(model, state, 0, prompt, params, noop, GenerationHooks.NONE);
            double ms = prefill ? r.promptMillis() : r.predictedMillis();
            double t = count / (ms / 1000.0);
            if (i >= warmup) tps[i - warmup] = t;
            System.err.printf("  %-6s [%-6s %2d] %8.2f t/s%n", test, i < warmup ? "warmup" : "rep", i, t);
        }
        return new Row(name, threads, test, mean(tps), stddev(tps));
    }

    /** Exactly {@code count} tokens of neutral filler (no chat template) — matches llama-bench's raw token count. */
    private static List<Integer> fillerTokens(Model model, int count) {
        StringBuilder sb = new StringBuilder();
        List<Integer> all;
        do {
            sb.append("The quick brown fox jumps over the lazy dog. ");
            all = model.tokenizer().encodeWithSpecialTokens(sb.toString());
        } while (all.size() < count);
        return new ArrayList<>(all.subList(0, count));
    }

    private record Row(String model, int threads, String test, double mean, double stddev) {}

    private static void printTable(List<Row> rows) {
        int w = rows.stream().mapToInt(r -> r.model.length()).max().orElse(5);
        w = Math.max(w, "model".length());
        String fmt = "| %-" + w + "s | %7s | %-6s | %16s |%n";
        System.out.printf(fmt, "model", "threads", "test", "t/s");
        System.out.printf(fmt, "-".repeat(w), "------:", "------", "---------------:");
        for (Row r : rows) {
            System.out.printf(fmt, r.model, r.threads, r.test, String.format("%.2f ± %.2f", r.mean, r.stddev));
        }
    }

    private static double mean(double[] a) { double s = 0; for (double v : a) s += v; return s / a.length; }

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
        out.println("""
            jinfer-bench — pp/tg throughput benchmark (llama-bench compatible)

            usage: jinfer-bench -m <model.gguf> [-m ...] [options]
              -m, --model <path>      model to benchmark (repeatable)
              -p, --n-prompt <N>      prompt/prefill tokens (default 512; 0 to skip pp)
              -n, --n-gen <N>         tokens to generate    (default 128; 0 to skip tg)
              -r, --repetitions <N>   timed repetitions      (default 5)
              -w, --warmup <N>        warmup passes          (default 2)
                  --ctx <N>           context size           (default p + n + 64)
              -h, --help

            threads: prefill uses the ForkJoinPool common pool; set with
              -Djava.util.concurrent.ForkJoinPool.common.parallelism=<N> and JAM_NUM_THREADS=<N>
              (and pin with taskset) to match a llama-bench -t <N> run.""");
    }
}
