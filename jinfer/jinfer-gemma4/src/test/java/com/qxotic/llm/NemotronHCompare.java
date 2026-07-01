// Diff the new com.qxotic.llm.NemotronH port against the production com.qxotic.jinfer Nemotron on the
// same tokens: prefill both, compare the next-token argmax + max per-logit diff, then 8 decode steps.
//   java ... com.qxotic.llm.NemotronHCompare <model.gguf> [prompt]
package com.qxotic.llm;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.FloatTensor;
import com.qxotic.jinfer.ModelLegacy;
import com.qxotic.jinfer.ModelLoader;

import java.nio.file.Path;
import java.util.List;

public final class NemotronHCompare {
    public static void main(String[] args) throws Exception {
        String path = args[0];
        String prompt = args.length > 1 ? args[1] : "The capital of France is";

        NemotronH nw = NemotronH.loadModel(Path.of(path), 4096);
        var tk = nw.tokenizer();
        List<Integer> pt = tk.encode(prompt);   // add_bos=false
        int[] ids = pt.stream().mapToInt(Integer::intValue).toArray();
        int vocab = nw.config().vocabularySize();

        ModelLegacy old = ModelLoader.loadModel(Path.of(path), 4096);
        var os = old.createNewState();
        old.ingest(os, ids, 0, 0, ids.length);
        FloatTensor ol = old.computeLogits(os);

        NemotronH.State ns = nw.newState(4096, Math.max(16, ids.length));
        nw.ingest(ns, Batch.prefill(ids));
        FloatTensor nl = nw.logits(ns);

        int oTok = LLM.argmax(ol, vocab), nTok = LLM.argmax(nl, vocab);
        double maxDiff = 0; int arg = 0;
        for (int i = 0; i < vocab; i++) {
            double d = Math.abs(ol.getFloat(i) - nl.getFloat(i));
            if (d > maxDiff) { maxDiff = d; arg = i; }
        }
        System.out.printf("prefill argmax: old=%d ('%s')  new=%d ('%s')  %s%n",
                oTok, tk.decode(oTok), nTok, tk.decode(nTok), oTok == nTok ? "MATCH" : "DIFFER");
        System.out.printf("max |logit diff| = %.5f @%d   (old[%d]=%.4f new=%.4f)%n",
                maxDiff, arg, oTok, ol.getFloat(oTok), nl.getFloat(oTok));

        int tok = oTok;
        for (int step = 0; step < 8; step++) {
            int pos = ids.length + step;
            old.ingest(os, new int[]{tok}, 0, pos, 1);
            FloatTensor ol2 = old.computeLogits(os);
            nw.ingest(ns, Batch.step(tok));
            FloatTensor nl2 = nw.logits(ns);
            int oT = LLM.argmax(ol2, vocab), nT = LLM.argmax(nl2, vocab);
            double md = 0;
            for (int i = 0; i < vocab; i++) { double d = Math.abs(ol2.getFloat(i) - nl2.getFloat(i)); if (d > md) md = d; }
            System.out.printf("decode %d (fed %d): old=%d('%s') new=%d('%s') %s  maxDiff=%.4f%n",
                    step, tok, oT, tk.decode(oT), nT, tk.decode(nT), oT == nT ? "MATCH" : "DIFFER", md);
            tok = oT;
        }
    }

}
