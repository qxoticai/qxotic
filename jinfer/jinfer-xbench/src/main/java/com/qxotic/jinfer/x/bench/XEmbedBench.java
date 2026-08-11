package com.qxotic.jinfer.x.bench;

import com.qxotic.jinfer.x.Segments;
import com.qxotic.jinfer.x.Views;
import java.io.PrintStream;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ForkJoinPool;

/**
 * Throughput A/B for the ragged/packed batched-embedding path ({@code EmbeddingModel.embed}), old
 * Lfm2 vs the x port IN THE SAME JVM behind {@code --impl old,x}: many variable-length sequences
 * packed into segmented forwards over one KV context, each sequence's pooled vector streamed out.
 * The workload is {@code EmbedBench}'s verbatim (ragged lengths by multiplicative hash, greedy
 * content-independent filler tokens), the timing loop is {@code XJinferBench}'s (adaptive warmup
 * until throughput settles, then timed reps) — one loop measures both trees, so JVM state is shared
 * and the comparison is the number, not the noise.
 *
 * <p>{@code --family} picks the model pair: {@code lfm2} (default) needs a bidirectional embedder
 * (pooling CLS, attention.causal=false — LFM2.5-ColBERT-350M works: dense_2 head, bidirectional;
 * both trees refuse causal checkpoints by name), {@code qwen3} a causal one (pooling LAST —
 * Qwen3-Embedding-0.6B; the reranker checkpoint shares the backbone).
 *
 * <pre>
 *   java ... XEmbedBench -m model.gguf [--family lfm2|qwen3] [--impl old,x] [-s 256] [--minlen 8] [--maxlen 64] [-b 512] [-r 5] [-w 3]
 * </pre>
 */
public final class XEmbedBench {

    private static volatile double sink; // blackhole so pooled vectors survive DCE

    public static void main(String[] args) throws Exception {
        String modelPath = null;
        String family = "lfm2";
        List<String> impls = new ArrayList<>();
        int nSeq = 256, minLen = 8, maxLen = 64, batchCap = 512, reps = 5, warmup = 3;
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "-m", "--model" -> modelPath = args[++i];
                case "--family" -> family = args[++i];
                case "--impl" -> impls.addAll(List.of(args[++i].split(",")));
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
        if (!family.equals("lfm2") && !family.equals("qwen3")) {
            System.err.println("unknown family: " + family);
            usage(System.err);
            System.exit(2);
        }
        if (impls.isEmpty()) impls = List.of("old", "x");

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
        List<String[]> rows = new ArrayList<>();
        for (String implName : impls) {
            System.err.printf(
                    "loading %s (ctx=%d; %d packed tokens across %d seqs, avg %.1f, batchCap=%d)"
                            + " impl=%s ...%n",
                    modelPath, ctx, total, nSeq, (double) total / nSeq, batchCap, implName);
            BenchModel impl = load(Path.of(modelPath), family, implName);
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
                System.err.printf("  embed [%s warmup %2d] %9.0f tok/s%n", implName, passes, t);
                if (passes >= Math.max(warmup, WINDOW)) {
                    double lo = Double.MAX_VALUE, hi = 0;
                    for (double v : recent) {
                        lo = Math.min(lo, v);
                        hi = Math.max(hi, v);
                    }
                    if ((hi - lo) / lo < TOL) break;
                }
            }
            System.err.printf("  embed [%s] stabilized after %d passes%n", implName, passes);

            double[] tps = new double[reps];
            for (int i = 0; i < reps; i++) {
                tps[i] = runOnce(impl, ids, seqLen, total, nSeq);
                System.err.printf("  embed [%s rep %2d] %9.0f tok/s%n", implName, i, tps[i]);
            }
            double meanTok = mean(tps), sd = stddev(tps);
            double meanSeq = meanTok * nSeq / total;
            rows.add(
                    new String[] {
                        name + " [" + implName + "]",
                        String.format("%.0f ± %.0f", meanTok, sd),
                        String.format("%.1f", meanSeq)
                    });
        }

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
    private static double runOnce(BenchModel impl, int[] ids, int[] seqLen, int total, int nSeq) {
        long t0 = System.nanoTime();
        int got = impl.embed(ids, seqLen);
        double tps = total / ((System.nanoTime() - t0) / 1e9);
        if (got != nSeq) {
            throw new IllegalStateException("expected " + nSeq + " embeddings, got " + got);
        }
        return tps;
    }

    /**
     * One impl = load + state + one packed embed pass. The seam is stateful like XJinferBench's:
     * both sides drive their CLAIMING public {@code embed} (the old interface default, the x
     * bidirectional override), which resets/reuses the state per call.
     */
    abstract static class BenchModel {
        abstract int vocab();

        abstract void newState(int ctx, int batchCap);

        /** Embeds the packed ragged batch; returns how many vectors streamed out. */
        abstract int embed(int[] ids, int[] seqLen);
    }

    static final class OldImpl extends BenchModel {
        private final com.qxotic.jinfer.models.lfm2.Lfm2 model;
        private com.qxotic.jinfer.models.lfm2.Lfm2.State s;

        OldImpl(Path path) throws Exception {
            this.model = com.qxotic.jinfer.models.lfm2.Lfm2.loadModel(path, Arena.ofAuto());
        }

        @Override
        int vocab() {
            return model.config().vocabularySize();
        }

        @Override
        void newState(int ctx, int batchCap) {
            s = model.newState(ctx, batchCap); // owned arena, GC/Cleaner-freed — as EmbedBench
        }

        @Override
        int embed(int[] ids, int[] seqLen) {
            int[] got = {0};
            model.embed(
                    s,
                    new com.qxotic.jinfer.Batch.Input.Sequences(
                            new com.qxotic.jinfer.Batch.Input.Tokens(ids), seqLen),
                    e -> {
                        sink += e.getFloat(0);
                        got[0]++;
                    });
            return got[0];
        }
    }

    static final class XImpl extends BenchModel {
        private final com.qxotic.jinfer.x.models.lfm2.Lfm2 model;
        private com.qxotic.jinfer.x.models.lfm2.Lfm2.State s;

        XImpl(Path path) throws Exception {
            this.model = com.qxotic.jinfer.x.models.lfm2.Lfm2.loadModel(path, Arena.ofAuto());
        }

        @Override
        int vocab() {
            return model.config().vocabularySize();
        }

        @Override
        void newState(int ctx, int batchCap) {
            s = model.newState(ctx, batchCap); // owned arena, GC/Cleaner-freed — as the old side
        }

        @Override
        int embed(int[] ids, int[] seqLen) {
            int[] got = {0};
            model.embed(
                    s,
                    new com.qxotic.jinfer.x.boundary.Batch.Input.Sequences(
                            new com.qxotic.jinfer.x.boundary.Batch.Input.Tokens(ids), seqLen),
                    e -> {
                        Views.Raw r = Views.rawF32(e, "embedding");
                        sink += Segments.readFloat(r.vseg(), r.vbase());
                        got[0]++;
                    });
            return got[0];
        }
    }

    static final class OldQwen3Impl extends BenchModel {
        private final com.qxotic.jinfer.models.qwen3.Qwen3 model;
        private com.qxotic.jinfer.models.qwen3.Qwen3.State s;

        OldQwen3Impl(Path path) throws Exception {
            this.model = com.qxotic.jinfer.models.qwen3.Qwen3.loadModel(path, Arena.ofAuto());
        }

        @Override
        int vocab() {
            return model.config().vocabularySize();
        }

        @Override
        void newState(int ctx, int batchCap) {
            s = model.newState(ctx, batchCap);
        }

        @Override
        int embed(int[] ids, int[] seqLen) {
            int[] got = {0};
            model.embed(
                    s,
                    new com.qxotic.jinfer.Batch.Input.Sequences(
                            new com.qxotic.jinfer.Batch.Input.Tokens(ids), seqLen),
                    e -> {
                        sink += e.getFloat(0);
                        got[0]++;
                    });
            return got[0];
        }
    }

    static final class XQwen3Impl extends BenchModel {
        private final com.qxotic.jinfer.x.models.qwen3.Qwen3 model;
        private com.qxotic.jinfer.x.models.qwen3.Qwen3.State s;

        XQwen3Impl(Path path) throws Exception {
            this.model = com.qxotic.jinfer.x.models.qwen3.Qwen3.loadModel(path, Arena.ofAuto());
        }

        @Override
        int vocab() {
            return model.config().vocabularySize();
        }

        @Override
        void newState(int ctx, int batchCap) {
            s = model.newState(ctx, batchCap);
        }

        @Override
        int embed(int[] ids, int[] seqLen) {
            int[] got = {0};
            model.embed(
                    s,
                    new com.qxotic.jinfer.x.boundary.Batch.Input.Sequences(
                            new com.qxotic.jinfer.x.boundary.Batch.Input.Tokens(ids), seqLen),
                    e -> {
                        Views.Raw r = Views.rawF32(e, "embedding");
                        sink += Segments.readFloat(r.vseg(), r.vbase());
                        got[0]++;
                    });
            return got[0];
        }
    }

    private static BenchModel load(Path path, String family, String impl) throws Exception {
        return switch (family + ":" + impl) {
            case "lfm2:old" -> new OldImpl(path);
            case "lfm2:x" -> new XImpl(path);
            case "qwen3:old" -> new OldQwen3Impl(path);
            case "qwen3:x" -> new XQwen3Impl(path);
            default ->
                    throw new IllegalArgumentException(
                            "unknown family/impl: " + family + ":" + impl);
        };
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
                XEmbedBench — ragged/packed batched-embedding throughput, old vs x (one JVM)

                usage: XEmbedBench -m <embedder.gguf> [options]
                  -m, --model <path>      embedding checkpoint (see --family)
                      --family <name>     lfm2: bidirectional (pooling CLS, attention.causal=false);
                                          qwen3: causal (pooling LAST). Default lfm2
                      --impl <list>       comma-separated impls to measure (default old,x)
                  -s, --sequences <N>     number of packed sequences (default 256)
                      --minlen <N>        min sequence length (default 8)
                      --maxlen <N>        max sequence length (default 64)
                  -b, --batch <N>         per-chunk forward width / batchCapacity (default 512)
                  -r, --repetitions <N>   timed reps (default 5)
                  -w, --warmup <N>        min warmup passes; warms adaptively until throughput settles (default 3)\
                """);
    }
}
