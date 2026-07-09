// Qwen3-Embedding runner: text (or explicit -Dtokens=..) -> embedding vector on stdout
// (space-separated),
// stats on stderr. -Dwarm=N runs N warmup passes first. Used to cosine-compare against llama.cpp.
package com.qxotic.jinfer.models.qwen35;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.FloatTensor;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

public final class Qwen3EmbedRun {
    public static void main(String[] args) throws Exception {
        int ctx = 2048;
        Qwen3 model = Qwen3.loadModel(Path.of(args[0]), ctx);
        var tk = model.tokenizer();

        int[] ids;
        int synth = Integer.getInteger("synth", 0);
        String tokprop = System.getProperty("tokens");
        if (synth > 0) { // -Dsynth=N synthetic tokens, for prefill benchmarking
            int vocab = model.config().vocabularySize();
            ids = new int[synth];
            for (int i = 0; i < synth; i++) ids[i] = (i * 17 + 1) % vocab;
        } else if (tokprop != null && !tokprop.isBlank()) {
            ids =
                    Arrays.stream(tokprop.split(","))
                            .map(String::trim)
                            .mapToInt(Integer::parseInt)
                            .toArray();
        } else {
            String text =
                    args.length > 1 ? args[1] : "The quick brown fox jumps over the lazy dog.";
            List<Integer> t = new java.util.ArrayList<>(tk.encode(text));
            t.add(
                    tk.getSpecialTokens()
                            .getOrDefault(
                                    "<|endoftext|>",
                                    151643)); // last-token pooling: append EOS, like llama.cpp
            ids = t.stream().mapToInt(Integer::intValue).toArray();
        }
        System.err.println("tokens(" + ids.length + "): " + Arrays.toString(ids));

        int warm = Integer.getInteger("warm", 0);
        for (int i = 0; i < warm; i++) {
            var w = model.newState(Math.max(16, ids.length), Math.max(16, ids.length));
            model.ingest(w, Batch.prefill(ids));
            model.embedding(w);
        }

        int bench = Integer.getInteger("bench", 0);
        if (bench > 0) { // -Dbench=N: time N warm passes, report prefill throughput
            int scap = Math.max(16, ids.length);
            boolean reuse =
                    Boolean.getBoolean(
                            "reuse"); // -Dreuse: reuse one state (reset position) vs alloc per
            // embed
            var rs = reuse ? model.newState(scap, scap) : null;
            long t0 = System.nanoTime();
            for (int i = 0; i < bench; i++) {
                var b = reuse ? rs : model.newState(scap, scap);
                if (reuse) b.position = 0; // reset the cursor; the prefill overwrites the KV ring
                model.ingest(b, Batch.prefill(ids));
                model.embedding(b);
            }
            double msPer = (System.nanoTime() - t0) / 1e6 / bench;
            System.err.printf(
                    "bench: %d tokens  %.3f ms/embed  %.1f tok/s  (warm=%d reps=%d reuse=%b)%n",
                    ids.length, msPer, ids.length * 1000.0 / msPer, warm, bench, reuse);
            return;
        }

        var s = model.newState(Math.max(16, ids.length), Math.max(16, ids.length));
        model.ingest(s, Batch.prefill(ids));
        FloatTensor emb = model.embedding(s);
        int dim = model.config().embeddingLength();

        StringBuilder sb = new StringBuilder(dim * 12);
        for (int i = 0; i < dim; i++) {
            if (i > 0) sb.append(' ');
            sb.append(emb.getFloat(i));
        }
        System.out.println(sb);
        System.err.printf(
                "dim=%d  emb[0..5]= %.6f %.6f %.6f %.6f %.6f%n",
                dim,
                emb.getFloat(0),
                emb.getFloat(1),
                emb.getFloat(2),
                emb.getFloat(3),
                emb.getFloat(4));
    }
}
