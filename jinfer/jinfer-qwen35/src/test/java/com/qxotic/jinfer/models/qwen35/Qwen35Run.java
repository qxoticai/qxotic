// Greedy decode smoke runner for the Qwen3.5 port.   java ...
// com.qxotic.jinfer.models.qwen35.Qwen35Run <model.gguf>
// [prompt] [nTokens]
package com.qxotic.jinfer.models.qwen35;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.llm.SpecialTokens;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

public final class Qwen35Run {
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
        String path = args[0];
        String promptStr = args.length > 1 ? args[1] : "The capital of France is";
        int nTokens = args.length > 2 ? Integer.parseInt(args[2]) : 32;

        Qwen35 model = Qwen35.loadModel(Path.of(path), 4096);
        var c = model.config();
        System.err.printf(
                "config: dim=%d layers=%d heads=%d kvHeads=%d vocab=%d ctx=%d experts=%d"
                        + " hidden=%d%n",
                c.embeddingLength,
                c.numberOfLayers,
                c.numberOfHeads,
                c.numberOfKeyValueHeads,
                c.vocabularySize(),
                c.contextLength(),
                c.expertCount,
                c.hiddenDim);

        var tk = model.tokenizer();
        List<Integer> promptTokens = new ArrayList<>(); // Qwen3.5 has no BOS
        if (System.getenv("CHAT")
                != null) { // ChatML: <|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n
            int imStart = SpecialTokens.find(tk, "<|im_start|>").orElse(-1);
            int imEnd = SpecialTokens.find(tk, "<|im_end|>").orElse(-1);
            if (imStart >= 0) promptTokens.add(imStart);
            promptTokens.addAll(tk.encode("user\n" + promptStr.strip()).toList());
            if (imEnd >= 0) promptTokens.add(imEnd);
            promptTokens.addAll(tk.encode("\n").toList());
            if (imStart >= 0) promptTokens.add(imStart);
            promptTokens.addAll(tk.encode("assistant\n").toList());
        } else {
            promptTokens.addAll(tk.encode(promptStr).toList());
        }
        int[] ids = promptTokens.stream().mapToInt(Integer::intValue).toArray();
        System.err.println("prompt tokens: " + promptTokens);

        Qwen35.State s = model.newState(c.contextLength(), Math.max(16, ids.length));
        model.ingest(s, Batch.prefill(ids));

        Set<Integer> stops = model.stopTokens();
        StringBuilder out = new StringBuilder();
        int tok = model.logits(s).argmax();
        int n = 0;
        long t0 = System.nanoTime();
        for (; n < nTokens && !stops.contains(tok); n++) {
            out.append(tk.decode(new int[] {tok}));
            model.ingest(s, Batch.step(tok));
            tok = model.logits(s).argmax();
        }
        double secs = (System.nanoTime() - t0) / 1e9;
        System.out.println("=== continuation ===");
        System.out.println(promptStr + out);
        System.err.printf("%n%.2f tok/s (%d tokens)%n", n / secs, n);
    }
}
