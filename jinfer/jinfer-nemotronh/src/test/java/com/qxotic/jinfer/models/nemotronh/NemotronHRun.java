// Greedy decode smoke runner for the NemotronH port.  java ...
// com.qxotic.jinfer.models.nemotronh.NemotronHRun
// <model.gguf> [prompt] [nTokens]
package com.qxotic.jinfer.models.nemotronh;

import com.qxotic.jinfer.Batch;
import java.nio.file.Path;
import java.util.List;
import java.util.Set;

public final class NemotronHRun {
    public static void main(String[] args) throws Exception {
        String path = args[0];
        String promptStr = args.length > 1 ? args[1] : "The capital of France is";
        int nTokens = args.length > 2 ? Integer.parseInt(args[2]) : 32;

        NemotronH model = NemotronH.loadModel(Path.of(path), 4096);
        var c = model.config();
        System.err.printf(
                "config: dim=%d layers=%d heads=%d kvHeads=%d vocab=%d ctx=%d experts=%d%n",
                c.embeddingLength(),
                c.numberOfLayers(),
                c.numberOfHeads(),
                c.numberOfKeyValueHeads(),
                c.vocabularySize(),
                c.contextLength(),
                c.expertCount());

        var tk = model.tokenizer();
        List<Integer> pt = tk.encode(promptStr).toList(); // add_bos=false: no leading BOS
        int[] ids = pt.stream().mapToInt(Integer::intValue).toArray();
        System.err.println("prompt tokens: " + pt);

        NemotronH.State s = model.newState(c.contextLength(), Math.max(16, ids.length));
        model.ingest(s, Batch.prefill(ids));

        Set<Integer> stops = model.stopTokens();
        StringBuilder out = new StringBuilder();
        int tok = model.logits(s).argmax();
        int n = 0;
        long t0 = System.nanoTime();
        for (; n < nTokens && !stops.contains(tok); n++) {
            out.append(tk.decode(tok));
            model.ingest(s, Batch.step(tok));
            tok = model.logits(s).argmax();
        }
        double secs = (System.nanoTime() - t0) / 1e9;
        System.out.println("=== continuation ===");
        System.out.println(promptStr + out);
        System.err.printf("%n%.2f tok/s (%d tokens)%n", n / secs, n);
    }
}
