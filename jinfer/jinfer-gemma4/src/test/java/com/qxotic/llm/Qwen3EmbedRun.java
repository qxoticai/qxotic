// Qwen3-Embedding runner: text (or explicit -Dtokens=..) -> embedding vector on stdout (space-separated),
// stats on stderr. -Dwarm=N runs N warmup passes first. Used to cosine-compare against llama.cpp.
package com.qxotic.llm;

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
        String tokprop = System.getProperty("tokens");
        if (tokprop != null && !tokprop.isBlank()) {
            ids = Arrays.stream(tokprop.split(",")).map(String::trim).mapToInt(Integer::parseInt).toArray();
        } else {
            String text = args.length > 1 ? args[1] : "The quick brown fox jumps over the lazy dog.";
            List<Integer> t = new java.util.ArrayList<>(tk.encode(text));
            t.add(tk.getSpecialTokens().getOrDefault("<|endoftext|>", 151643));   // last-token pooling: append EOS, like llama.cpp
            ids = t.stream().mapToInt(Integer::intValue).toArray();
        }
        System.err.println("tokens(" + ids.length + "): " + Arrays.toString(ids));

        int warm = Integer.getInteger("warm", 0);
        for (int i = 0; i < warm; i++) {
            var w = model.newState(ctx, Math.max(16, ids.length));
            model.ingest(w, Batch.prefill(ids));
            model.embedding(w);
        }

        var s = model.newState(ctx, Math.max(16, ids.length));
        model.ingest(s, Batch.prefill(ids));
        FloatTensor emb = model.embedding(s);
        int dim = model.config().embeddingLength();

        StringBuilder sb = new StringBuilder(dim * 12);
        for (int i = 0; i < dim; i++) { if (i > 0) sb.append(' '); sb.append(emb.getFloat(i)); }
        System.out.println(sb);
        System.err.printf("dim=%d  emb[0..5]= %.6f %.6f %.6f %.6f %.6f%n",
                dim, emb.getFloat(0), emb.getFloat(1), emb.getFloat(2), emb.getFloat(3), emb.getFloat(4));
    }
}
