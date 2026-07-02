package com.qxotic.jinfer;

import java.nio.file.Path;
import java.util.List;

/** Diagnostic: ingest a prompt (batched + single-token) and report argmax + NaN/Inf in logits,
 *  to isolate why prompts containing special tokens produce garbage. */
public final class DiagSpecial {
    static void probe(Model model, String label, List<Integer> ids, boolean single) {
        InferenceState state = model.createNewState();
        int[] toks = ids.stream().mapToInt(Integer::intValue).toArray();
        if (single) for (int i = 0; i < toks.length; i++) model.ingest(state, toks, i, i, 1);
        else model.ingest(state, toks, 0, 0, toks.length);
        FloatTensor logits = model.computeLogits(state);
        int n = model.vocabularySize();
        int nan = 0, argmax = 0; float best = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < n; i++) {
            float v = logits.getFloat(i);
            if (Float.isNaN(v) || Float.isInfinite(v)) nan++;
            if (v > best) { best = v; argmax = i; }
        }
        System.out.printf("%-22s single=%-5s argmax=%-8d best=%-12.4f NaN/Inf=%d%n",
                label, single, argmax, best, nan);
    }

    public static void main(String[] args) throws Exception {
        Model model = LegacyModelLoader.loadModel(Path.of(args[0]), 4096);
        var tok = model.tokenizer();
        List<Integer> chatml = tok.encodeWithSpecialTokens(
                "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n");
        List<Integer> plain = tok.encodeWithSpecialTokens("What is the capital of France?");
        probe(model, "plain", plain, false);
        probe(model, "plain", plain, true);
        probe(model, "chatml", chatml, false);
        probe(model, "chatml", chatml, true);
        // single special token alone
        Integer imStart = tok.getSpecialTokens().get("<|im_start|>");
        probe(model, "<|im_start|> only", List.of(imStart), false);
    }
}
