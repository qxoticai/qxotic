// PURE-ENGINE probe (no MTP code): does a state whose KV was written by multi-row score batches
// produce different next-token logits than a state advanced by single-step decode over the SAME
// tokens? This isolates the batch-vs-step KV numerics the MTP verify loop rides on. If the swaps
// and margins here match the MtpLockstepOracle divergences, the speculative loop is exonerated -
// the divergence is a pre-existing engine property of mixing chunk shapes.
//   java ... com.qxotic.jinfer.models.gemma4.BatchVsStepProbe [chunk]
package com.qxotic.jinfer.models.gemma4;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.FloatTensor;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

public final class BatchVsStepProbe {

    public static void main(String[] args) throws Exception {
        int chunk = args.length > 0 ? Integer.parseInt(args[0]) : 3; // = MTP depth+1
        Path model =
                Path.of("/home/mukel/Desktop/playground/models/unsloth/gemma-4-E2B-it-Q8_0.gguf");
        if (!Files.exists(model)) {
            System.out.println("BatchVsStepProbe: model absent, skipping");
            return;
        }
        Gemma4 m = Gemma4.loadModel(model, 4096);
        var tk = m.tokenizer();
        int bos = tk.getSpecialTokens().getOrDefault("<bos>", 2);
        Set<Integer> stops = m.stopTokens();
        int vocab = m.config().vocabularySize();

        int totalSwaps = 0;
        double maxRel = 0;
        for (String prompt :
                new String[] {
                    "The capital of France is",
                    "Write a short poem about the sea.",
                    "def fibonacci(n):",
                    "List the first ten prime numbers:",
                    "Once upon a time, in a quiet village by the mountains,"
                }) {
            int[] ids = withBos(bos, tk.encode(prompt));

            // reference: plain greedy, single-step KV
            Gemma4.State a = m.newState(4096, Math.max(16, ids.length));
            m.ingest(a, Batch.prefill(ids));
            List<Integer> toks = new ArrayList<>();
            int t = m.logits(a, 0).argmax(0, vocab);
            while (toks.size() < 120 && !stops.contains(t)) {
                toks.add(t);
                m.ingest(a, Batch.step(t));
                t = m.logits(a, 0).argmax(0, vocab);
            }

            // batch-KV replay: same tokens ingested as score-chunks; compare each chunk's row
            // logits
            Gemma4.State b = m.newState(4096, Math.max(16, ids.length));
            m.ingest(b, Batch.prefill(ids));
            // oracle single-step state advanced alongside for per-position reference logits
            Gemma4.State o = m.newState(4096, Math.max(16, ids.length));
            m.ingest(o, Batch.prefill(ids));

            int swaps = 0;
            StringBuilder detail = new StringBuilder();
            boolean reject = Boolean.getBoolean("probe.rejectPattern");
            for (int i = 0; i < toks.size(); i += (reject ? 1 : chunk)) {
                int n =
                        reject
                                ? Math.min(chunk, toks.size() - i)
                                : Math.min(chunk, toks.size() - i);
                int[] c = new int[n];
                for (int j = 0; j < n; j++) c[j] = toks.get(i + j);
                if (reject
                        && !Boolean.getBoolean(
                                "probe.correctDrafts")) { // MTP 0-accept pattern: junk drafts
                    for (int j = 1; j < n; j++) c[j] = 1000 + 997 * j; // deliberately wrong drafts
                } // with -Dprobe.correctDrafts=true: drafts are the TRUE next tokens, still rolled
                // back
                long base = b.position();
                m.ingest(b, Batch.score(c));
                int keepRows = reject ? 1 : n;
                for (int j = 0; j < keepRows; j++) {
                    FloatTensor bl = m.logits(b, j);
                    int bAm = bl.argmax(0, vocab);
                    float bTop = bl.getFloat(bAm);
                    m.ingest(o, Batch.step(toks.get(i + j)));
                    FloatTensor ol = m.logits(o, 0);
                    int oAm = ol.argmax(0, vocab);
                    if (bAm != oAm) {
                        swaps++;
                        double rel =
                                Math.abs(ol.getFloat(oAm) - ol.getFloat(bAm))
                                        / Math.max(1e-6, Math.abs(ol.getFloat(oAm)));
                        maxRel = Math.max(maxRel, rel);
                        if (detail.length() < 1500) {
                            detail.append(
                                    String.format(
                                            "    pos+%d: batchArgmax=%d(%s) stepArgmax=%d(%s)"
                                                    + " stepRel=%.2e batchTop=%.3f%n",
                                            i + j,
                                            bAm,
                                            esc(tk.decode(bAm)),
                                            oAm,
                                            esc(tk.decode(oAm)),
                                            rel,
                                            bTop));
                        }
                    }
                }
                if (reject) b.resumeAt((int) base + 1); // roll back the junk drafts
            }
            totalSwaps += swaps;
            System.out.printf(
                    "%-38s tokens=%3d argmax-swaps=%d%n",
                    "\"" + head(prompt) + "\"", toks.size(), swaps);
            if (swaps > 0) System.out.print(detail);
        }
        System.out.printf(
                "TOTAL argmax swaps (batch-row vs step, same tokens)=%d maxStepRel=%.3e%n",
                totalSwaps, maxRel);
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
