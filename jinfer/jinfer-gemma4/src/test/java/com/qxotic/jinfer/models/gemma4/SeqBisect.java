package com.qxotic.jinfer.models.gemma4;

import com.qxotic.jinfer.FloatTensor;
import com.qxotic.jinfer.llm.SpecialTokens;
import java.nio.file.Path;
import java.util.Arrays;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Diagnostic: for each seqLen, compare the last-token logits from a single batched prefill against
 * ingesting the same tokens one at a time. The two are mathematically identical (causal attention),
 * so any gross divergence isolates a seqLen-dependent bug in the batched (multi-row) GEMM path.
 *
 * <pre>SeqBisect &lt;model.gguf&gt; [maxSeq]</pre>
 */
public final class SeqBisect {
    @Test
    @Tag("driver")
    void run() throws Exception {
        Assumptions.assumeTrue(
                !System.getProperty("jinfer.args", "").isBlank(),
                "set -Djinfer.args=\"...\" to run this tool");
        main(testArgs());
    }

    private static String[] testArgs() {
        String argv = System.getProperty("jinfer.args", "");
        return argv.isBlank() ? new String[0] : argv.trim().split("\\s+");
    }

    private static void main(String[] args) throws Exception {
        Gemma4 model = Gemma4.loadModel(Path.of(args[0]), 4096);
        int vocab = model.config().vocabularySize();
        int maxSeq = args.length > 1 ? Integer.parseInt(args[1]) : 24;
        int[] filler = filler(model, maxSeq);

        for (int seqLen = 1; seqLen <= maxSeq; seqLen++) {
            int[] toks = Arrays.copyOf(filler, seqLen);

            Gemma4.State sb = model.newState(256, Math.max(16, seqLen));
            model.ingest(sb, com.qxotic.jinfer.Batch.prefill(toks));
            float[] batched = snapshot(model.logits(sb), vocab);

            Gemma4.State ss = model.newState(256, 16);
            for (int t : toks) model.ingest(ss, com.qxotic.jinfer.Batch.step(t));
            float[] step = snapshot(model.logits(ss), vocab);

            double maxAbs = 0;
            for (int i = 0; i < vocab; i++)
                maxAbs = Math.max(maxAbs, Math.abs(batched[i] - step[i]));
            int ab = argmax(batched), as = argmax(step);
            System.out.printf(
                    "seqLen=%2d  argmax batched=%-7d step=%-7d %s  maxAbsDiff=%.4g%n",
                    seqLen, ab, as, ab == as ? "OK      " : "MISMATCH", maxAbs);
        }
    }

    private static float[] snapshot(FloatTensor t, int n) {
        float[] r = new float[n];
        for (int i = 0; i < n; i++) r[i] = t.getFloat(i);
        return r;
    }

    private static int argmax(float[] x) {
        int b = 0;
        for (int i = 1; i < x.length; i++) if (x[i] > x[b]) b = i;
        return b;
    }

    private static int[] filler(Gemma4 model, int n) {
        StringBuilder sb = new StringBuilder();
        com.qxotic.toknroll.IntSequence all;
        do {
            sb.append("The quick brown fox jumps over the lazy dog. ");
            all = SpecialTokens.encode(model.tokenizer(), sb.toString());
        } while (all.length() < n);
        int[] ids = new int[n];
        for (int i = 0; i < n; i++) ids[i] = all.intAt(i);
        return ids;
    }
}
