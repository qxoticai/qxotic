// Runnable smoke/parity check for the jinfer-gemma4 port: load a real Gemma GGUF and greedily
// decode through the new com.qxotic.jinfer.Model seam (newState → ingest(prefill) → logits →
// ingest(step)).
//   java ... com.qxotic.jinfer.models.gemma4.GemmaRun [model.gguf] [prompt] [nTokens]
package com.qxotic.jinfer.models.gemma4;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

public final class GemmaRun {
    public static void main(String[] args) throws Exception {
        String path =
                args.length > 0
                        ? args[0]
                        : "/home/mukel/Desktop/playground/models/google/gemma-4-E2B_q4_0-it.gguf";
        String promptStr = args.length > 1 ? args[1] : "The capital of France is";
        int nTokens = args.length > 2 ? Integer.parseInt(args[2]) : 20;
        int contextCapacity =
                args.length > 3 ? Integer.parseInt(args[3]) : -1; // -1 → full model context

        Gemma4 model = Gemma4.loadModel(Path.of(path), 4096);
        var c = model.config();
        System.err.printf(
                "config: dim=%d layers=%d heads=%d vocab=%d ctx=%d ownKv=%d plDim=%d%n",
                c.embeddingLength(),
                c.numberOfLayers(),
                c.numberOfHeads(),
                c.vocabularySize(),
                c.contextLength(),
                c.ownKvLayers(),
                c.embeddingLengthPerLayer());

        var tk = model.tokenizer();
        int bos = tk.getSpecialTokens().getOrDefault("<bos>", 2);
        List<Integer> promptTokens = new ArrayList<>();
        promptTokens.add(bos);
        promptTokens.addAll(tk.encode(promptStr));
        int[] ids = promptTokens.stream().mapToInt(Integer::intValue).toArray();
        System.err.println("prompt tokens: " + promptTokens);

        int cap = contextCapacity > 0 ? contextCapacity : c.contextLength();
        Gemma4.State s = model.newState(cap, Math.max(16, ids.length));
        System.err.println("contextCapacity=" + cap);
        long t0 = System.nanoTime();
        if (System.getenv("STEP_PREFILL") != null) { // ingest the prompt one token at a time
            for (int id : ids) model.ingest(s, com.qxotic.jinfer.Batch.step(id));
        } else {
            model.ingest(s, com.qxotic.jinfer.Batch.prefill(ids));
        }

        Set<Integer> stops = model.stopTokens();
        StringBuilder out = new StringBuilder();
        int tok = model.logits(s).argmax();
        int n = 0;
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
