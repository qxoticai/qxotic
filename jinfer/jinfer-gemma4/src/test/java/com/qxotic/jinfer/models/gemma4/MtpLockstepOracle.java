// Divergence diagnostic for the MTP speculative loop: run the spec loop with the top-2 recorder,
// then advance a plain single-step oracle over the SAME emitted stream and, at every emission where
// the oracle disagrees, print BOTH sides' top-2 margins. Near-tie engine noise shows as: the two
// paths swap top1/top2 with tiny margins on both sides. A real loop bug shows as a confident
// disagreement (large margins, disjoint tokens).
//   java [-Djinfer.disableJam=true] [-Djava.util.concurrent.ForkJoinPool.common.parallelism=1] ...
//        com.qxotic.jinfer.models.gemma4.MtpLockstepOracle [depth]
package com.qxotic.jinfer.models.gemma4;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.FloatTensor;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

public final class MtpLockstepOracle {

    static final String[] PROMPTS = {
        "The capital of France is",
        "Write a short poem about the sea.",
        "def fibonacci(n):",
        "List the first ten prime numbers:",
        "Once upon a time, in a quiet village by the mountains,",
    };

    record Emit(int token, int top1, float l1, int top2, float l2) {}

    public static void main(String[] args) throws Exception {
        int depth = args.length > 0 ? Integer.parseInt(args[0]) : 2;
        Path model =
                Path.of("/home/mukel/Desktop/playground/models/unsloth/gemma-4-E2B-it-Q8_0.gguf");
        Path sidecar =
                Path.of("/home/mukel/Desktop/playground/models/unsloth/mtp-gemma-4-E2B-it.gguf");
        if (!Files.exists(model) || !Files.exists(sidecar)) {
            System.out.println("MtpLockstepOracle: model/sidecar absent, skipping");
            return;
        }
        Gemma4 m = Gemma4.loadModel(model, 4096, sidecar);
        var tk = m.tokenizer();
        int bos = tk.getSpecialTokens().getOrDefault("<bos>", 2);
        Set<Integer> stops = m.stopTokens();
        int vocab = m.config().vocabularySize();

        System.out.printf(
                "config: depth=%d disableJam=%s parallelism=%s%n",
                depth,
                System.getProperty("jinfer.disableJam", "false"),
                System.getProperty(
                        "java.util.concurrent.ForkJoinPool.common.parallelism", "default"));

        int totalDiv = 0, nearTie = 0, confident = 0;
        double maxOracleRel = 0;
        for (String prompt : PROMPTS) {
            int[] ids = withBos(bos, tk.encode(prompt).toList());

            // spec run with recorder
            List<Emit> emits = new ArrayList<>();
            Gemma4.State ss = m.newState(4096, Math.max(16, ids.length));
            m.ingest(ss, Batch.prefill(ids));
            Gemma4Speculative.generate(
                    m,
                    ss,
                    120,
                    stops,
                    depth,
                    (t, t1, l1, t2, l2) -> emits.add(new Emit(t, t1, l1, t2, l2)));

            // repeat run: is the spec output reproducible with identical config?
            List<Integer> emitsB = new ArrayList<>();
            Gemma4.State sb = m.newState(4096, Math.max(16, ids.length));
            m.ingest(sb, Batch.prefill(ids));
            Gemma4Speculative.generate(
                    m, sb, 120, stops, depth, (t, t1, l1, t2, l2) -> emitsB.add(t));
            int rep = 0;
            while (rep < Math.min(emits.size(), emitsB.size())
                    && emits.get(rep).token == emitsB.get(rep)) rep++;
            boolean reproducible = rep == emits.size() && emits.size() == emitsB.size();

            // lockstep oracle: single-step decode over the SAME emitted stream
            Gemma4.State os = m.newState(4096, Math.max(16, ids.length));
            m.ingest(os, Batch.prefill(ids));
            int div = 0;
            StringBuilder detail = new StringBuilder();
            for (Emit e : emits) {
                FloatTensor ol = m.logits(os, os.outputCount() - 1);
                int oam = ol.argmax(0, vocab);
                if (oam != e.token) {
                    div++;
                    float lEmit = ol.getFloat(e.token), lOam = ol.getFloat(oam);
                    double oracleRel = Math.abs(lOam - lEmit) / Math.max(1e-6, Math.abs(lOam));
                    double specRel = Math.abs(e.l1 - e.l2) / Math.max(1e-6, Math.abs(e.l1));
                    boolean swap = e.top2 == oam; // the two paths merely swapped top1/top2
                    maxOracleRel = Math.max(maxOracleRel, oracleRel);
                    if (swap && (oracleRel < 2e-2 || specRel < 2e-2)) nearTie++;
                    else confident++;
                    if (detail.length() < 2000) {
                        detail.append(
                                String.format(
                                        "    div@%d emit=%d(%s) oracleTop1=%d(%s) oracleRel=%.2e"
                                                + " specTop2=%d specRel=%.2e swap=%b%n",
                                        emits.indexOf(e),
                                        e.token,
                                        esc(tk.decode(e.token)),
                                        oam,
                                        esc(tk.decode(oam)),
                                        oracleRel,
                                        e.top2,
                                        specRel,
                                        swap));
                    }
                }
                m.ingest(os, Batch.step(e.token));
            }
            totalDiv += div;
            System.out.printf(
                    "%-38s emits=%3d div=%d reproducible=%b (rep to %d)%n",
                    "\"" + head(prompt) + "\"", emits.size(), div, reproducible, rep);
            if (div > 0) System.out.print(detail);
        }
        System.out.printf(
                "TOTAL divergences=%d nearTie=%d confident=%d maxOracleRel=%.3e%n",
                totalDiv, nearTie, confident, maxOracleRel);
    }

    static int[] withBos(int bos, List<Integer> enc) {
        int[] ids = new int[enc.size() + 1];
        ids[0] = bos;
        for (int i = 0; i < enc.size(); i++) ids[i + 1] = enc.get(i);
        return ids;
    }

    static String esc(String s) {
        return s.replace("\n", "\\n");
    }

    static String head(String s) {
        return s.length() <= 30 ? s : s.substring(0, 30) + "...";
    }
}
