// Runnable smoke/parity check for the LFM2.5 port: load a real LFM2.5 GGUF and greedily decode
// through the com.qxotic.llm.Model seam.   java ... com.qxotic.llm.Lfm2Run [model.gguf] [prompt] [nTokens]
package com.qxotic.llm;

import com.qxotic.jinfer.FloatTensor;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

public final class Lfm2Run {
    public static void main(String[] args) throws Exception {
        String path = args[0];
        String promptStr = args.length > 1 ? args[1] : "The capital of France is";
        int nTokens = args.length > 2 ? Integer.parseInt(args[2]) : 32;

        Lfm2 model = Lfm2.loadModel(Path.of(path), 4096);
        var c = model.config();
        System.err.printf("config: dim=%d layers=%d heads=%d vocab=%d ctx=%d dConv=%d experts=%d%n",
                c.embeddingLength(), c.numberOfLayers(), c.numberOfHeads(), c.vocabularySize(), c.contextLength(),
                c.shortConvLCache(), c.expertCount());

        var tk = model.tokenizer();
        int bos = tk.getSpecialTokens().getOrDefault("<bos>", 1);
        List<Integer> promptTokens = new ArrayList<>();
        promptTokens.add(bos);
        if (System.getenv("CHAT") != null) {   // LFM2 ChatML: <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
            int imStart = tk.getSpecialTokens().getOrDefault("<|im_start|>", bos);
            int imEnd = tk.getSpecialTokens().getOrDefault("<|im_end|>", -1);
            promptTokens.add(imStart);
            promptTokens.addAll(tk.encode("user\n" + promptStr.strip()));
            if (imEnd >= 0) promptTokens.add(imEnd);
            promptTokens.addAll(tk.encode("\n"));
            promptTokens.add(imStart);
            promptTokens.addAll(tk.encode("assistant\n"));
        } else {
            promptTokens.addAll(tk.encode(promptStr));
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
