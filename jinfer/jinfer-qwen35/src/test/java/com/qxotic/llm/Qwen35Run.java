// Greedy decode smoke runner for the Qwen3.5 port.   java ... com.qxotic.llm.Qwen35Run <model.gguf> [prompt] [nTokens]
package com.qxotic.llm;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.FloatTensor;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

public final class Qwen35Run {
    public static void main(String[] args) throws Exception {
        String path = args[0];
        String promptStr = args.length > 1 ? args[1] : "The capital of France is";
        int nTokens = args.length > 2 ? Integer.parseInt(args[2]) : 32;

        Qwen35 model = Qwen35.loadModel(Path.of(path), 4096);
        var c = model.config();
        System.err.printf("config: dim=%d layers=%d heads=%d kvHeads=%d vocab=%d ctx=%d experts=%d hidden=%d%n",
                c.embeddingLength, c.numberOfLayers, c.numberOfHeads, c.numberOfKeyValueHeads, c.vocabularySize(),
                c.contextLength(), c.expertCount, c.hiddenDim);

        var tk = model.tokenizer();
        List<Integer> promptTokens = new ArrayList<>();   // Qwen3.5 has no BOS
        if (System.getenv("CHAT") != null) {              // ChatML: <|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n
            int imStart = tk.getSpecialTokens().getOrDefault("<|im_start|>", -1);
            int imEnd = tk.getSpecialTokens().getOrDefault("<|im_end|>", -1);
            if (imStart >= 0) promptTokens.add(imStart);
            promptTokens.addAll(tk.encode("user\n" + promptStr.strip()));
            if (imEnd >= 0) promptTokens.add(imEnd);
            promptTokens.addAll(tk.encode("\n"));
            if (imStart >= 0) promptTokens.add(imStart);
            promptTokens.addAll(tk.encode("assistant\n"));
        } else {
            promptTokens.addAll(tk.encode(promptStr));
        }
        int[] ids = promptTokens.stream().mapToInt(Integer::intValue).toArray();
        System.err.println("prompt tokens: " + promptTokens);

        Qwen35.State s = model.newState(c.contextLength(), Math.max(16, ids.length));
        model.ingest(s, Batch.prefill(ids));

        Set<Integer> stops = model.stopTokens();
        StringBuilder out = new StringBuilder();
        int tok = LLM.argmax(model.logits(s), c.vocabularySize());
        int n = 0;
        long t0 = System.nanoTime();
        for (; n < nTokens && !stops.contains(tok); n++) {
            out.append(tk.decode(tok));
            model.ingest(s, Batch.step(tok));
            tok = LLM.argmax(model.logits(s), c.vocabularySize());
        }
        double secs = (System.nanoTime() - t0) / 1e9;
        System.out.println("=== continuation ===");
        System.out.println(promptStr + out);
        System.err.printf("%n%.2f tok/s (%d tokens)%n", n / secs, n);
    }

}
