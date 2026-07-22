// Runnable smoke/parity check for the LFM2.5 port: load a real LFM2.5 GGUF and greedily decode
// through the com.qxotic.jinfer.Model seam.   java ... com.qxotic.jinfer.models.lfm2.Lfm2Run
// [model.gguf] [prompt]
// [nTokens]
package com.qxotic.jinfer.models.lfm2;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

public final class Lfm2Run {
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

        Lfm2 model = Lfm2.loadModel(Path.of(path), 4096);
        var c = model.config();
        System.err.printf(
                "config: dim=%d layers=%d heads=%d vocab=%d ctx=%d dConv=%d experts=%d%n",
                c.embeddingLength(),
                c.numberOfLayers(),
                c.numberOfHeads(),
                c.vocabularySize(),
                c.contextLength(),
                c.shortConvLCache(),
                c.expertCount());

        var tk = model.tokenizer();
        int bos = com.qxotic.jinfer.llm.SpecialTokens.find(tk, "<bos>").orElse(1);
        List<Integer> promptTokens = new ArrayList<>();
        promptTokens.add(bos);
        if (System.getenv("CHAT") != null) { // LFM2 ChatML:
            // <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
            int imStart = com.qxotic.jinfer.llm.SpecialTokens.find(tk, "<|im_start|>").orElse(bos);
            int imEnd = com.qxotic.jinfer.llm.SpecialTokens.find(tk, "<|im_end|>").orElse(-1);
            promptTokens.add(imStart);
            promptTokens.addAll(tk.encode("user\n" + promptStr.strip()).toList());
            if (imEnd >= 0) promptTokens.add(imEnd);
            promptTokens.addAll(tk.encode("\n").toList());
            promptTokens.add(imStart);
            promptTokens.addAll(tk.encode("assistant\n").toList());
        } else {
            promptTokens.addAll(tk.encode(promptStr).toList());
        }
        int[] ids = promptTokens.stream().mapToInt(Integer::intValue).toArray();
        System.err.println("prompt tokens: " + promptTokens);

        Lfm2.State s = model.newState(c.contextLength(), Math.max(16, ids.length));
        model.ingest(s, com.qxotic.jinfer.Batch.prefill(ids));

        Set<Integer> stops = model.stopTokens();
        StringBuilder out = new StringBuilder();
        int tok = model.logits(s).argmax();
        int n = 0;
        long t0 = System.nanoTime();
        for (; n < nTokens && !stops.contains(tok); n++) {
            out.append(tk.decode(new int[] {tok}));
            model.ingest(s, com.qxotic.jinfer.Batch.step(tok));
            tok = model.logits(s).argmax();
        }
        double secs = (System.nanoTime() - t0) / 1e9;
        System.out.println("=== continuation ===");
        System.out.println(promptStr + out);
        System.err.printf("%n%.2f tok/s (%d tokens)%n", n / secs, n);
    }
}
