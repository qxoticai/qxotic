// Runnable smoke check for the standard-Llama port: load a Llama-family GGUF and greedily decode
// through
// the com.qxotic.jinfer.models model API.   java ... com.qxotic.jinfer.models.llama.LlamaRun
// <model.gguf> [prompt] [nTokens]
// CHAT=1 wraps the prompt in the Llama-3 header chat format.
package com.qxotic.jinfer.models.llama;

import com.qxotic.jinfer.llm.SpecialTokens;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

public final class LlamaRun {
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

        Llama model = Llama.loadModel(Path.of(path), 4096);
        var c = model.config();
        System.err.printf(
                "config: dim=%d layers=%d heads=%d kvHeads=%d vocab=%d ctx=%d%n",
                c.embeddingLength(),
                c.numberOfLayers(),
                c.numberOfHeads(),
                c.numberOfKeyValueHeads(),
                c.vocabularySize(),
                c.contextLength());

        var tk = model.tokenizer();
        int bos =
                SpecialTokens.find(tk, "<bos>")
                        .orElse(
                                SpecialTokens.find(tk, "<|begin_of_text|>")
                                        .orElse(
                                                SpecialTokens.find(tk, "<|startoftext|>")
                                                        .orElse(1)));
        List<Integer> promptTokens = new ArrayList<>();
        if (model.config().addBos()) promptTokens.add(bos);
        if (System.getenv("CHAT") != null) { // Llama-3:
            // <|start_header_id|>user<|end_header_id|>\n\n{p}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
            int sh = SpecialTokens.find(tk, "<|start_header_id|>").orElse(-1);
            int eh = SpecialTokens.find(tk, "<|end_header_id|>").orElse(-1);
            int eot = SpecialTokens.find(tk, "<|eot_id|>").orElse(-1);
            if (sh >= 0 && eh >= 0 && eot >= 0) {
                promptTokens.add(sh);
                promptTokens.addAll(tk.encode("user").toList());
                promptTokens.add(eh);
                promptTokens.addAll(tk.encode("\n\n" + promptStr.strip()).toList());
                promptTokens.add(eot);
                promptTokens.add(sh);
                promptTokens.addAll(tk.encode("assistant").toList());
                promptTokens.add(eh);
                promptTokens.addAll(tk.encode("\n\n").toList());
            } else {
                promptTokens.addAll(tk.encode(promptStr).toList());
            }
        } else {
            promptTokens.addAll(tk.encode(promptStr).toList());
        }
        int[] ids = promptTokens.stream().mapToInt(Integer::intValue).toArray();
        System.err.println("prompt tokens: " + promptTokens);

        Llama.State s = model.newState(c.contextLength(), Math.max(16, ids.length));
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
