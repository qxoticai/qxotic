// Qwen3 packed batched-embedding: isolation invariant (packed == individual) + throughput vs
// one-by-one.
//   correctness: java ... Qwen3BatchRun <model.gguf>            [-Dbc=N to force chunking]
//   benchmark:   java ... Qwen3BatchRun <model.gguf> -Dbench=NSEQ -Dlen=L -Dwarm=W -Dreps=R
package com.qxotic.jinfer.models.qwen35;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.FloatTensor;
import com.qxotic.jinfer.llm.SpecialTokens;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

public final class Qwen3BatchRun {
    @Test
    @Tag("driver")
    void run() throws Exception {
        Assumptions.assumeTrue(
                !System.getProperty("jinfer.args", "").isBlank(),
                "set -Djinfer.args=\"<model.gguf> ...\" to run this tool");
        main(testArgs());
    }

    private static String[] testArgs() {
        String argv = System.getProperty("jinfer.args", "");
        return argv.isBlank() ? new String[0] : argv.trim().split("\\s+");
    }

    private static void main(String[] args) throws Exception {
        int ctx = 8192;
        Qwen3 model = Qwen3.loadModel(Path.of(args[0]), ctx);
        var tk = model.tokenizer();
        int eos = SpecialTokens.find(tk, "<|endoftext|>").orElse(151643);
        int dim = model.config().embeddingLength();
        int vocab = model.config().vocabularySize();

        int bench = Integer.getInteger("bench", 0);
        if (bench > 0) {
            benchmark(model, dim, vocab, eos, bench);
            return;
        }

        String promptsFile = System.getProperty("prompts");
        if (promptsFile
                != null) { // batched-embed each line, print vectors (space-separated) for llama.cpp
            // cross-check
            List<String> lines = java.nio.file.Files.readAllLines(Path.of(promptsFile));
            int[][] qs = new int[lines.size()][];
            for (int i = 0; i < lines.size(); i++) {
                List<Integer> t = new ArrayList<>(tk.encode(lines.get(i)).toList());
                t.add(eos);
                qs[i] = t.stream().mapToInt(Integer::intValue).toArray();
            }
            int tot = 0;
            for (int[] q : qs) tot += q.length;
            var st = model.newState(tot, tot);
            var pk = (Batch.Input.Sequences) Batch.pack(qs).input();
            StringBuilder sb = new StringBuilder();
            model.embed(
                    st,
                    pk,
                    e -> {
                        for (int i = 0; i < dim; i++) {
                            if (i > 0) sb.append(' ');
                            sb.append(e.getFloat(i));
                        }
                        sb.append('\n');
                    });
            System.out.print(sb);
            return;
        }

        // --- correctness: a batch of mixed-length real texts ---
        String[] texts = {
            "hello",
            "The cat sat on the mat.",
            "Paris is the capital of France.",
            "Machine learning is a subset of artificial intelligence that learns patterns from"
                    + " data.",
            "def add(a, b):\n    return a + b",
            "Photosynthesis converts light energy into chemical energy stored in glucose within"
                    + " plant cells.",
            "yes",
            "The quick brown fox jumps over the lazy dog while the sun sets behind the distant"
                    + " hills.",
        };
        int[][] seqs = new int[texts.length][];
        for (int i = 0; i < texts.length; i++) {
            List<Integer> t = new ArrayList<>(tk.encode(texts[i]).toList());
            t.add(eos);
            seqs[i] = t.stream().mapToInt(Integer::intValue).toArray();
        }
        int total = 0;
        for (int[] q : seqs) total += q.length;

        // individual (reference single path)
        float[][] indiv = new float[seqs.length][];
        for (int i = 0; i < seqs.length; i++) {
            var s = model.newState(seqs[i].length, seqs[i].length);
            model.ingest(s, Batch.prefill(seqs[i]));
            indiv[i] = toArray(model.embedding(s), dim);
        }

        // batched (packed); -Dbc lets us force chunking (default: single chunk)
        int bc = Integer.getInteger("bc", total);
        var bs = model.newState(total, bc);
        var packed = (Batch.Input.Sequences) Batch.pack(seqs).input();
        List<float[]> batched = new ArrayList<>();
        model.embed(bs, packed, e -> batched.add(toArray(e, dim)));

        // each sequence embedded as its OWN 1-sequence pack (segmented path, same GEMM N as
        // individual)
        float[][] solo = new float[seqs.length][];
        for (int i = 0; i < seqs.length; i++) {
            var ss = model.newState(seqs[i].length, seqs[i].length);
            var p1 = (Batch.Input.Sequences) Batch.pack(new int[][] {seqs[i]}).input();
            List<float[]> out = new ArrayList<>();
            model.embed(ss, p1, e -> out.add(toArray(e, dim)));
            solo[i] = out.get(0);
        }

        System.out.printf(
                "=== isolation invariant: packed (bc=%d, total=%d) vs individual ===%n", bc, total);
        double worst = 1.0, worstSolo = 1.0;
        for (int i = 0; i < seqs.length; i++) {
            double c = cosine(indiv[i], batched.get(i)), cs1 = cosine(indiv[i], solo[i]);
            worst = Math.min(worst, c);
            worstSolo = Math.min(worstSolo, cs1);
            System.out.printf(
                    "  seq %-2d (%3d tok): packed=%.6f  solo(same-N)=%.6f  %s%n",
                    i, seqs[i].length, c, cs1, c >= 0.9999 ? "OK" : "(<.9999)");
        }
        System.out.printf("worst packed=%.6f  worst solo(same-N)=%.6f%n", worst, worstSolo);
    }

    static void benchmark(Qwen3 model, int dim, int vocab, int eos, int nSeq) {
        int len = Integer.getInteger("len", 32),
                warm = Integer.getInteger("warm", 8),
                reps = Integer.getInteger("reps", 20);
        int[][] seqs = new int[nSeq][];
        for (int i = 0; i < nSeq; i++) {
            int[] q = new int[len];
            for (int j = 0; j < len - 1; j++) q[j] = ((i * 131 + j * 17 + 3) % (vocab - 10));
            q[len - 1] = eos;
            seqs[i] = q;
        }
        int total = nSeq * len;
        var packed = (Batch.Input.Sequences) Batch.pack(seqs).input();

        // batched: one reused state, whole packed batch per rep
        var bs = model.newState(total, total);
        Runnable batched =
                () -> {
                    model.embed(bs, packed, e -> {});
                };
        // one-by-one: reuse one right-sized state, reset per seq
        var os = model.newState(len, len);
        Runnable onebyone =
                () -> {
                    for (int[] q : seqs) {
                        os.reset();
                        model.ingest(os, Batch.prefill(q));
                        model.embedding(os);
                    }
                };

        double msB = timed(batched, warm, reps), msO = timed(onebyone, warm, reps);
        System.out.printf(
                "=== batched embedding throughput (%d seqs x %d tok = %d tokens, warm JVM) ===%n",
                nSeq, len, total);
        System.out.printf(
                "  batched:    %.2f ms  ->  %.1f emb/s   %.0f tok/s%n",
                msB, nSeq * 1000.0 / msB, total * 1000.0 / msB);
        System.out.printf(
                "  one-by-one: %.2f ms  ->  %.1f emb/s   %.0f tok/s%n",
                msO, nSeq * 1000.0 / msO, total * 1000.0 / msO);
        System.out.printf("  speedup:    %.2fx%n", msO / msB);
    }

    static double timed(Runnable r, int warm, int reps) {
        for (int i = 0; i < warm; i++) r.run();
        long t0 = System.nanoTime();
        for (int i = 0; i < reps; i++) r.run();
        return (System.nanoTime() - t0) / 1e6 / reps;
    }

    static float[] toArray(FloatTensor t, int dim) {
        float[] a = new float[dim];
        for (int i = 0; i < dim; i++) a[i] = t.getFloat(i);
        return a;
    }

    static double cosine(float[] a, float[] b) {
        double d = 0, na = 0, nb = 0;
        for (int i = 0; i < a.length; i++) {
            d += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        return d / (Math.sqrt(na) * Math.sqrt(nb) + 1e-12);
    }
}
