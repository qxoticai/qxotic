package com.qxotic.jinfer;

import com.qxotic.llm.Qwen3;

import java.io.PrintStream;
import java.nio.file.Path;
import java.util.concurrent.ForkJoinPool;

/**
 * Throughput benchmark for the ragged/packed batched-embedding path ({@link EmbeddingModel#embed}): many
 * variable-length sequences packed into a single segmented forward over one KV context, each sequence's
 * pooled vector streamed out. Reports tokens/s and sequences/s, warmed adaptively like {@link JinferBench}
 * (the JVM must reach JIT/GC steady state before timing). Greedy, content-independent filler tokens.
 *
 * <pre>embed-bench -m Qwen3-Embedding-*.gguf [-s 256] [--minlen 8] [--maxlen 64] [-b 512] [-r 5] [-w 3]</pre>
 */
public final class EmbedBench {

    private static volatile double sink;   // blackhole so the pooled vectors aren't dead-code-eliminated

    public static void main(String[] args) throws Exception {
        String modelPath = null;
        int nSeq = 256, minLen = 8, maxLen = 64, batchCap = 512, reps = 5, warmup = 3;
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "-m", "--model" -> modelPath = args[++i];
                case "-s", "--sequences" -> nSeq = Integer.parseInt(args[++i]);
                case "--minlen" -> minLen = Integer.parseInt(args[++i]);
                case "--maxlen" -> maxLen = Integer.parseInt(args[++i]);
                case "-b", "--batch" -> batchCap = Integer.parseInt(args[++i]);
                case "-r", "--repetitions" -> reps = Integer.parseInt(args[++i]);
                case "-w", "--warmup" -> warmup = Integer.parseInt(args[++i]);
                case "-h", "--help" -> { usage(System.out); return; }
                default -> { System.err.println("unknown option: " + args[i]); usage(System.err); System.exit(2); }
            }
        }
        if (modelPath == null) { usage(System.err); System.exit(2); }

        // Ragged lengths in [minLen, maxLen] (deterministic pseudo-random via a multiplicative hash).
        int[] seqLen = new int[nSeq];
        int total = 0, span = Math.max(1, maxLen - minLen + 1);
        for (int j = 0; j < nSeq; j++) {
            seqLen[j] = minLen + (int) ((j * 2654435761L & 0x7fffffffL) % span);
            total += seqLen[j];
        }
        int ctx = total + 64;   // the whole packed stream must fit in one context

        System.err.printf("loading %s (ctx=%d; %d packed tokens across %d seqs, avg %.1f, batchCap=%d) ...%n",
                modelPath, ctx, total, nSeq, (double) total / nSeq, batchCap);
        Qwen3 model = Qwen3.loadModel(Path.of(modelPath), ctx);
        int vocab = model.config().vocabularySize();
        int[] ids = new int[total];
        for (int i = 0; i < total; i++) ids[i] = (i * 17 + 1) % vocab;
        Batch.Input.Sequences seqs = new Batch.Input.Sequences(new Batch.Input.Tokens(ids), seqLen);

        Qwen3.State state = model.newState(ctx, batchCap);   // embed() resets it each call, so reuse it
        int threads = ForkJoinPool.commonPool().getParallelism();

        // Adaptive warmup: run until the last WINDOW throughputs agree within TOL (JIT/GC settled).
        final double TOL = 0.03;
        final int WINDOW = 3, MAX = Math.max(warmup, 30);
        double[] recent = new double[WINDOW];
        int passes = 0;
        while (passes < MAX) {
            double t = runOnce(model, state, seqs, total, nSeq);
            recent[passes % WINDOW] = t;
            passes++;
            System.err.printf("  embed [warmup %2d] %9.0f tok/s%n", passes, t);
            if (passes >= Math.max(warmup, WINDOW)) {
                double lo = Double.MAX_VALUE, hi = 0;
                for (double v : recent) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
                if ((hi - lo) / lo < TOL) break;
            }
        }
        System.err.printf("  embed stabilized after %d warmup passes%n", passes);

        double[] tps = new double[reps];
        for (int i = 0; i < reps; i++) {
            tps[i] = runOnce(model, state, seqs, total, nSeq);
            System.err.printf("  embed [rep    %2d] %9.0f tok/s%n", i, tps[i]);
        }

        double meanTok = mean(tps), sd = stddev(tps);
        double meanSeq = meanTok * nSeq / total;   // seqs/s from tok/s and the packed ratio
        String name = Path.of(modelPath).getFileName().toString().replaceAll("\\.gguf$", "");
        int w = Math.max(name.length(), "model".length());
        String fmt = "| %-" + w + "s | %7s | %5s | %14s | %11s |%n";
        System.out.printf(fmt, "model", "threads", "seqs", "tok/s", "seq/s");
        System.out.printf(fmt, "-".repeat(w), "------:", "----:", "-------------:", "----------:");
        System.out.printf(fmt, name, threads, nSeq,
                String.format("%.0f ± %.0f", meanTok, sd), String.format("%.1f", meanSeq));
    }

    /** One packed-embedding pass over all {@code nSeq} sequences; returns tokens/second. */
    private static double runOnce(Qwen3 model, Qwen3.State state, Batch.Input.Sequences seqs, int total, int nSeq) {
        int[] got = {0};
        long t0 = System.nanoTime();
        model.embed(state, seqs, e -> { sink += e.getFloat(0); got[0]++; });
        double tps = total / ((System.nanoTime() - t0) / 1e9);
        if (got[0] != nSeq) throw new IllegalStateException("expected " + nSeq + " embeddings, got " + got[0]);
        return tps;
    }

    private static double mean(double[] a) { double s = 0; for (double v : a) s += v; return s / a.length; }

    private static double stddev(double[] a) {
        if (a.length < 2) return 0;
        double m = mean(a), s = 0;
        for (double v : a) s += (v - m) * (v - m);
        return Math.sqrt(s / (a.length - 1));
    }

    private static void usage(PrintStream out) {
        out.println("""
            embed-bench — ragged/packed batched-embedding throughput (EmbeddingModel.embed)

            usage: embed-bench -m <Qwen3-Embedding-*.gguf> [options]
              -m, --model <path>      embedding model to benchmark
              -s, --sequences <N>     number of packed sequences (default 256)
                  --minlen <N>        min sequence length (default 8)
                  --maxlen <N>        max sequence length (default 64)
              -b, --batch <N>         per-chunk forward width / batchCapacity (default 512)
              -r, --repetitions <N>   timed reps (default 5)
              -w, --warmup <N>        min warmup passes; warms adaptively until throughput settles (default 3)""");
    }
}
