// Runnable smoke check for the gpt-oss port: load a real gpt-oss GGUF and greedily decode through the
// new model API.   java ... com.qxotic.llm.GptOssRun <model.gguf> [prompt] [nTokens]
// Raw prompt only (gpt-oss's harmony chat format is not wired here); for correctness use GptOssCompare,
// which checks token-exactness against the production GptOss.
package com.qxotic.llm;

import com.qxotic.jinfer.FloatTensor;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

public final class GptOssRun {
    public static void main(String[] args) throws Exception {
        String path = args[0];
        String promptStr = args.length > 1 ? args[1] : "The capital of France is";
        int nTokens = args.length > 2 ? Integer.parseInt(args[2]) : 32;

        GptOss model = GptOss.loadModel(Path.of(path), 4096);
        var c = model.config();
        System.err.printf("config: dim=%d layers=%d heads=%d kvHeads=%d headSize=%d vocab=%d ctx=%d experts=%d/%d expertFF=%d swaWin=%d%n",
                c.embeddingLength(), c.numberOfLayers(), c.numberOfHeads(), c.numberOfKeyValueHeads(), c.headSize(),
                c.vocabularySize(), c.contextLength(), c.expertUsedCount(), c.expertCount(), c.expertFeedForwardLength(), c.slidingWindow());

        var tk = model.tokenizer();
        int bos = tk.getSpecialTokens().getOrDefault("<|startoftext|>", 199998);
        List<Integer> promptTokens = new ArrayList<>();
        promptTokens.add(bos);
        promptTokens.addAll(tk.encode(promptStr));
        int[] ids = promptTokens.stream().mapToInt(Integer::intValue).toArray();
        System.err.println("prompt tokens: " + promptTokens);

        GptOss.State s = model.newState(c.contextLength(), Math.max(16, ids.length));
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
