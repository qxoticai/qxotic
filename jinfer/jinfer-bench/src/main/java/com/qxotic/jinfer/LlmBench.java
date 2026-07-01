package com.qxotic.jinfer;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.llm.Gemma4;
import com.qxotic.llm.Granite;
import com.qxotic.llm.GptOss;
import com.qxotic.llm.Lfm2;
import com.qxotic.llm.Llama;
import com.qxotic.llm.NemotronH;
import com.qxotic.llm.Qwen35;

import java.io.PrintStream;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ForkJoinPool;

/**
 * pp/tg throughput benchmark for the new com.qxotic.llm seam (the jinfer-gemma4 port), printed in the
 * same markdown table as {@link JinferBench} so the new API's numbers are directly comparable to the
 * production engine. Drives the forward directly — {@code newState → ingest → logits} — and times it
 * with {@code nanoTime} (the seam has no internal timers). Greedy argmax (temp 0), like llama-bench.
 *
 * <pre>jinfer-bench-llm -m gemma-4-*.gguf [-p 512] [-n 128] [-r 5] [-w 2] [--ctx N]</pre>
 */
public final class LlmBench {

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
        if (ctx == 0) ctx = Math.max(p, 1) + n + 64;
        int threads = ForkJoinPool.commonPool().getParallelism();

        List<Row> rows = new ArrayList<>();
        for (String path : models) {
            System.err.printf("loading %s (ctx=%d) via com.qxotic.llm ...%n", path, ctx);
            LanguageModel<?, ?, ?> model = loadAny(Path.of(path), ctx);
            String name = name(path);
            if (p > 0) rows.add(measure(model, name, threads, "pp" + p, p, true, warmup, reps));
            if (n > 0) rows.add(measure(model, name, threads, "tg" + n, n, false, warmup, reps));
        }
        printTable(rows);
    }

    /** Dispatch on general.architecture to the matching new-API port (all implement LanguageModel). */
    private static LanguageModel<?, ?, ?> loadAny(Path path, int ctx) throws Exception {
        String arch;
        try (FileChannel fc = FileChannel.open(path, StandardOpenOption.READ)) {
            arch = ModelLoader.readGguf(fc, path.toString()).getString("general.architecture");
        }
        return switch (arch) {
            case "gemma4" -> Gemma4.loadModel(path, ctx);
            case "gpt-oss" -> GptOss.loadModel(path, ctx);
            case "qwen35", "qwen35moe" -> Qwen35.loadModel(path, ctx);
            case "nemotron_h", "nemotron_h_moe" -> NemotronH.loadModel(path, ctx);
            case "llama", "minicpm", "mistral3" -> Llama.loadModel(path, ctx);   // mistral3 is a same-graph Llama variant
            case "granite" -> Granite.loadModel(path, ctx);
            default -> {
                if (arch.startsWith("lfm")) yield Lfm2.loadModel(path, ctx);
                throw new IllegalArgumentException("LlmBench: unsupported architecture '" + arch + "'");
            }
        };
    }

    /** One pp/tg test on the new seam: warmup then {@code reps} timed passes; throughput mean ± stddev. */
    private static <S extends RuntimeState> Row measure(LanguageModel<?, ?, S> model, String name, int threads, String test, int count, boolean prefill,
                               int warmup, int reps) {
        int ctx = model.config().contextLength();
        int vocab = model.config().vocabularySize();
        int[] prompt = fillerTokens(vocab, prefill ? count : 1);

        double[] tps = new double[reps];
        for (int i = 0; i < warmup + reps; i++) {
            S s = model.newState(ctx, Math.max(prompt.length, 16));
            double t;
            if (prefill) {
                // pp: one batched prefill of `count` tokens
                long t0 = System.nanoTime();
                model.ingest(s, Batch.prefill(prompt));
                t = count / ((System.nanoTime() - t0) / 1e9);
            } else {
                // tg: prime with one token, then time `count` single-token decode steps
                model.ingest(s, Batch.prefill(prompt));
                int tok = argmax(model.logits(s), vocab);
                long t0 = System.nanoTime();
                for (int g = 0; g < count; g++) {
                    model.ingest(s, Batch.step(tok));
                    tok = argmax(model.logits(s), vocab);
                }
                t = count / ((System.nanoTime() - t0) / 1e9);
            }
            if (i >= warmup) tps[i - warmup] = t;
            System.err.printf("  %-6s [%-6s %2d] %8.2f t/s%n", test, i < warmup ? "warmup" : "rep", i, t);
        }
        return new Row(name, threads, test, mean(tps), stddev(tps));
    }

    /** Synthetic in-range token ids — throughput is content-independent, and tokenizer() isn't on the interface. */
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
        for (Row r : rows) System.out.printf(fmt, r.model, r.threads, r.test, String.format("%.2f ± %.2f", r.mean, r.stddev));
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
            jinfer-bench-llm — pp/tg throughput for the com.qxotic.llm seam (jinfer-gemma4)

            usage: LlmBench -m <gemma-4-*.gguf> [-m ...] [options]
              -m, --model <path>      model to benchmark (repeatable)
              -p, --n-prompt <N>      prefill tokens (default 512; 0 to skip pp)
              -n, --n-gen <N>         decode tokens  (default 128; 0 to skip tg)
              -r, --repetitions <N>   timed reps     (default 5)
              -w, --warmup <N>        warmup passes  (default 2)
                  --ctx <N>           context size   (default p + n + 64)""");
    }
}
