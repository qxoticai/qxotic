// Runnable smoke check for the standard-Llama port: load a Llama-family GGUF and greedily decode through
// the com.qxotic.llm model API.   java ... com.qxotic.llm.LlamaRun <model.gguf> [prompt] [nTokens]
// CHAT=1 wraps the prompt in the Llama-3 header chat format.
package com.qxotic.llm;

import com.qxotic.jinfer.FloatTensor;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

public final class LlamaRun {
    public static void main(String[] args) throws Exception {
        String path = args[0];
        String promptStr = args.length > 1 ? args[1] : "The capital of France is";
        int nTokens = args.length > 2 ? Integer.parseInt(args[2]) : 32;

        Llama model = Llama.loadModel(Path.of(path), 4096);
        var c = model.config();
        System.err.printf("config: dim=%d layers=%d heads=%d kvHeads=%d vocab=%d ctx=%d%n",
                c.embeddingLength(), c.numberOfLayers(), c.numberOfHeads(), c.numberOfKeyValueHeads(), c.vocabularySize(), c.contextLength());

        var tk = model.tokenizer();
        int bos = LlamaCompare.bos(tk);
        List<Integer> promptTokens = new ArrayList<>();
        promptTokens.add(bos);
        if (System.getenv("CHAT") != null) {   // Llama-3: <|start_header_id|>user<|end_header_id|>\n\n{p}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
            int sh = tk.getSpecialTokens().getOrDefault("<|start_header_id|>", -1);
            int eh = tk.getSpecialTokens().getOrDefault("<|end_header_id|>", -1);
            int eot = tk.getSpecialTokens().getOrDefault("<|eot_id|>", -1);
            if (sh >= 0 && eh >= 0 && eot >= 0) {
                promptTokens.add(sh); promptTokens.addAll(tk.encode("user")); promptTokens.add(eh);
                promptTokens.addAll(tk.encode("\n\n" + promptStr.strip())); promptTokens.add(eot);
                promptTokens.add(sh); promptTokens.addAll(tk.encode("assistant")); promptTokens.add(eh);
                promptTokens.addAll(tk.encode("\n\n"));
            } else {
                promptTokens.addAll(tk.encode(promptStr));
            }
        } else {
            promptTokens.addAll(tk.encode(promptStr));
        }
        int[] ids = promptTokens.stream().mapToInt(Integer::intValue).toArray();
        System.err.println("prompt tokens: " + promptTokens);

        Llama.State s = model.newState(c.contextLength(), Math.max(16, ids.length));
        model.ingest(s, com.qxotic.jinfer.Batch.prefill(ids));

        Set<Integer> stops = model.stopTokens();
        StringBuilder out = new StringBuilder();
        int tok = LLM.argmax(model.logits(s), c.vocabularySize());
        int n = 0;
        long t0 = System.nanoTime();
        for (; n < nTokens && !stops.contains(tok); n++) {
            out.append(tk.decode(tok));
            model.ingest(s, com.qxotic.jinfer.Batch.step(tok));
            tok = LLM.argmax(model.logits(s), c.vocabularySize());
        }
        double secs = (System.nanoTime() - t0) / 1e9;
        System.out.println("=== continuation ===");
        System.out.println(promptStr + out);
        System.err.printf("%n%.2f tok/s (%d tokens)%n", n / secs, n);
    }

}
