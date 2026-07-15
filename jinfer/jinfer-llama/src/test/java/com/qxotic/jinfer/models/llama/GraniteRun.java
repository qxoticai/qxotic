// Runnable smoke check for the Granite 4.1 port: load the granite GGUF and greedily decode through
// the
// com.qxotic.jinfer.models model API.   java ... com.qxotic.jinfer.models.llama.GraniteRun
// <model.gguf> [prompt] [nTokens]
package com.qxotic.jinfer.models.llama;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

public final class GraniteRun {
    public static void main(String[] args) throws Exception {
        String path = args[0];
        String promptStr = args.length > 1 ? args[1] : "The capital of France is";
        int nTokens = args.length > 2 ? Integer.parseInt(args[2]) : 32;

        Granite model = Granite.loadModel(Path.of(path), 4096);
        var c = model.config();
        System.err.printf(
                "config: dim=%d layers=%d heads=%d kvHeads=%d vocab=%d ctx=%d attnScale=%.5f%n",
                c.embeddingLength(),
                c.numberOfLayers(),
                c.numberOfHeads(),
                c.numberOfKeyValueHeads(),
                c.vocabularySize(),
                c.contextLength(),
                c.attentionScale());

        var tk = model.tokenizer();
        List<Integer> promptTokens = new ArrayList<>();
        if (c.addBos()) promptTokens.add(c.bosTokenId());
        promptTokens.addAll(tk.encode(promptStr).toList());
        int[] ids = promptTokens.stream().mapToInt(Integer::intValue).toArray();
        System.err.println("prompt tokens: " + promptTokens);

        Granite.State s = model.newState(c.contextLength(), Math.max(16, ids.length));
        model.ingest(s, com.qxotic.jinfer.Batch.prefill(ids));

        Set<Integer> stops = model.stopTokens();
        StringBuilder out = new StringBuilder();
        int tok = model.logits(s).argmax();
        int n = 0;
        long t0 = System.nanoTime();
        for (; n < nTokens && !stops.contains(tok); n++) {
            out.append(tk.decode(tok));
            model.ingest(s, com.qxotic.jinfer.Batch.step(tok));
            tok = model.logits(s).argmax();
        }
        double secs = (System.nanoTime() - t0) / 1e9;
        System.out.println("=== continuation ===");
        System.out.println(promptStr + out);
        System.err.printf("%n%.2f tok/s (%d tokens)%n", n / secs, n);
    }
}
