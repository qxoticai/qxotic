package com.qxotic.jinfer.x.bench;

import com.qxotic.jinfer.x.Views;
import com.qxotic.jinfer.x.boundary.Batch;
import com.qxotic.jinfer.x.boundary.ContextState;
import com.qxotic.jinfer.x.chat.LoadedEmbedder;
import com.qxotic.jinfer.x.chat.Models;
import java.io.PrintStream;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ForkJoinPool;

/**
 * Throughput for the ragged/packed batched-embedding path ({@code EmbeddingModel.embedAll}), any
 * embedder the x tree loads ({@link Models#loadEmbedder} - no per-architecture code): many
 * variable-length sequences packed into segmented forwards over one KV context, each sequence's
 * pooled vector streamed out. The workload is legacy {@code EmbedBench}'s verbatim (ragged lengths
 * by multiplicative hash, greedy content-independent filler tokens), the timing loop is {@code
 * XJinferBench}'s (adaptive warmup until throughput settles, then timed reps).
 *
 * <pre>
 *   java ... XEmbedBench -m model.gguf [-s 256] [--minlen 8] [--maxlen 64] [-b 512] [-r 5] [-w 3]
 * </pre>
 */
public final class XEmbedBench {

    private static volatile double sink; // blackhole so pooled vectors survive DCE

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
        if (modelPath == null) {
            usage(System.err);
            System.exit(2);
        }

        // Ragged lengths in [minLen, maxLen] (deterministic pseudo-random via a multiplicative
        // hash) - EmbedBench's exact workload.
        int[] seqLen = new int[nSeq];
        int total = 0, span = Math.max(1, maxLen - minLen + 1);
        for (int j = 0; j < nSeq; j++) {
            seqLen[j] = minLen + (int) ((j * 2654435761L & 0x7fffffffL) % span);
            total += seqLen[j];
        }
        int ctx = total + 64; // the whole packed stream must fit in one context
        int threads = ForkJoinPool.commonPool().getParallelism();

        String name = Path.of(modelPath).getFileName().toString().replaceAll("\\.gguf$", "");
        System.err.printf(
                "loading %s (ctx=%d; %d packed tokens across %d seqs, avg %.1f, batchCap=%d) ...%n",
                modelPath, ctx, total, nSeq, (double) total / nSeq, batchCap);
        EmbedModel<?> impl = EmbedModel.open(Path.of(modelPath));
        int vocab = impl.vocab();
        int[] ids = new int[total];
        for (int i = 0; i < total; i++) ids[i] = (i * 17 + 1) % vocab;
        impl.newState(ctx, batchCap);

        // Adaptive warmup: run until the last WINDOW throughputs agree within TOL.
        final double TOL = 0.03;
        final int WINDOW = 3, MAX = Math.max(warmup, 30);
        double[] recent = new double[WINDOW];
        int passes = 0;
        while (passes < MAX) {
            double t = runOnce(impl, ids, seqLen, total, nSeq);
            recent[passes % WINDOW] = t;
            passes++;
            System.err.printf("  embed [warmup %2d] %9.0f tok/s%n", passes, t);
            if (passes >= Math.max(warmup, WINDOW)) {
                double lo = Double.MAX_VALUE, hi = 0;
                for (double v : recent) {
                    lo = Math.min(lo, v);
                    hi = Math.max(hi, v);
                }
                if ((hi - lo) / lo < TOL) break;
            }
        }
        System.err.printf("  embed stabilized after %d passes%n", passes);

        double[] tps = new double[reps];
        for (int i = 0; i < reps; i++) {
            tps[i] = runOnce(impl, ids, seqLen, total, nSeq);
            System.err.printf("  embed [rep %2d] %9.0f tok/s%n", i, tps[i]);
        }
        double meanTok = mean(tps), sd = stddev(tps);
        double meanSeq = meanTok * nSeq / total;

        List<String[]> rows = new ArrayList<>();
        rows.add(
                new String[] {
                    name, String.format("%.0f ± %.0f", meanTok, sd), String.format("%.1f", meanSeq)
                });
        int w = rows.stream().mapToInt(r -> r[0].length()).max().orElse(5);
        w = Math.max(w, "model".length());
        String fmt = "| %-" + w + "s | %7s | %5s | %14s | %11s |%n";
        System.out.printf(fmt, "model", "threads", "seqs", "tok/s", "seq/s");
        System.out.printf(fmt, "-".repeat(w), "------:", "----:", "-------------:", "----------:");
        for (String[] row : rows) {
            System.out.printf(fmt, row[0], threads, nSeq, row[1], row[2]);
        }
    }

    /** One packed-embedding pass over all sequences; returns tokens/second. */
    private static double runOnce(
            EmbedModel<?> impl, int[] ids, int[] seqLen, int total, int nSeq) {
        long t0 = System.nanoTime();
        int got = impl.embed(ids, seqLen);
        double tps = total / ((System.nanoTime() - t0) / 1e9);
        if (got != nSeq) {
            throw new IllegalStateException("expected " + nSeq + " embeddings, got " + got);
        }
        return tps;
    }

    /**
     * The one model wrapper: whatever {@link Models#loadEmbedder} resolves, driven at the raw
     * boundary. The state is owned-arena, GC/Cleaner-freed, and embedAll resets/reuses it per call.
     */
    private static final class EmbedModel<S extends ContextState> {
        private final LoadedEmbedder<S> loaded;
        private S state;

        private EmbedModel(LoadedEmbedder<S> loaded) {
            this.loaded = loaded;
        }

        @SuppressWarnings("unchecked")
        static EmbedModel<?> open(Path path) throws java.io.IOException {
            return new EmbedModel<>(
                    (LoadedEmbedder<ContextState>) Models.loadEmbedder(path, Arena.ofAuto()));
        }

        int vocab() {
            return loaded.model().configuration().vocabularySize();
        }

        void newState(int ctx, int batchCap) {
            state = loaded.model().newState(ctx, batchCap);
        }

        /** Embeds the packed ragged batch; returns how many vectors streamed out. */
        int embed(int[] ids, int[] seqLen) {
            int[] got = {0};
            loaded.model()
                    .embedAll(
                            state,
                            new Batch.Input.Sequences(new Batch.Input.Tokens(ids), seqLen),
                            e -> {
                                sink +=
                                        Views.getFloat(
                                                Views.castToSegmentBacked(e, "embedding"),
                                                0,
                                                "embedding");
                                got[0]++;
                            });
            return got[0];
        }
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

    private static void usage(PrintStream out) {
        out.println(
                """
                XEmbedBench - ragged/packed batched-embedding throughput (x tree, generic loader)

                usage: XEmbedBench -m <embedder.gguf> [options]
                  -m, --model <path>      embedding checkpoint (bidirectional like LFM2.5-ColBERT,
                                          causal like Qwen3-Embedding - the port declares pooling)
                  -s, --sequences <N>     number of packed sequences (default 256)
                      --minlen <N>        min sequence length (default 8)
                      --maxlen <N>        max sequence length (default 64)
                  -b, --batch <N>         per-chunk forward width / batchCapacity (default 512)
                  -r, --repetitions <N>   timed reps (default 5)
                  -w, --warmup <N>        min warmup passes; warms adaptively until throughput settles (default 3)\
                """);
    }
}
