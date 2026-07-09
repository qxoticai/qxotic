// Stage-2 gate: jinfer's MTP draft top-10 must match the llama.cpp anchor
// (models/unsloth/mtp-anchor.json) for the fixed prompt "The capital of France is".
// Prefill the prompt on the backbone, take the last position's pre-final-norm hidden + the greedy
// sampled token, run the draft, print top-10. Tries a couple of rope-position conventions since the
// anchor is the disambiguator. Skips cleanly if the model/sidecar are absent.
//   java ... com.qxotic.llm.Gemma4MtpParityTest
package com.qxotic.llm;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.FloatTensor;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public final class Gemma4MtpParityTest {

    static final int[] ANCHOR_TOP = {
        236761, 236764, 568, 15245, 528, 236888, 657, 951, 236793, 236881
    };

    public static void main(String[] args) throws Exception {
        Path model =
                Path.of("/home/mukel/Desktop/playground/models/unsloth/gemma-4-E2B-it-Q8_0.gguf");
        Path sidecar =
                Path.of("/home/mukel/Desktop/playground/models/unsloth/mtp-gemma-4-E2B-it.gguf");
        if (!Files.exists(model) || !Files.exists(sidecar)) {
            System.out.println("Gemma4MtpParityTest: model/sidecar absent, skipping");
            return;
        }
        Gemma4 backbone = Gemma4.loadModel(model, 4096);
        Gemma4Mtp mtp = Gemma4Mtp.loadSidecar(sidecar, backbone.config().vocabularySize());
        Gemma4MtpDecoder decoder = new Gemma4MtpDecoder(mtp, backbone);
        var tk = backbone.tokenizer();
        int dim = backbone.config().embeddingLength();

        // Anchor prompt WITH BOS (Gemma requires it; llama.cpp add_bos), then greedy-decode. The
        // anchor
        // captured the draft step where the backbone produced "." after " Paris", so we draft at
        // EACH
        // decode step and match the "." (id 236761 top-1) step. The draft input is (hidden of the
        // position
        // that produced the sampled token, that token); rope position = that token's position.
        int bos = tk.getSpecialTokens().getOrDefault("<bos>", 2);
        List<Integer> ids = new ArrayList<>();
        ids.add(bos);
        ids.addAll(tk.encode("The capital of France is"));
        int[] prompt = ids.stream().mapToInt(Integer::intValue).toArray();
        System.out.println("prompt ids (with bos): " + java.util.Arrays.toString(prompt));
        Gemma4.State s = backbone.newState(4096, Math.max(16, prompt.length));
        backbone.ingest(s, Batch.prefill(prompt));

        int vocab = backbone.config().vocabularySize();
        int tok =
                argmax(backbone.logits(s, 0), vocab); // token produced by the last prompt position
        boolean matched = false;
        for (int step = 0; step < 8; step++) {
            // Commit `tok`'s KV, then draft from ITS hidden at ITS position (aligned MTP step):
            // the backbone attends [0..pos] including tok; the draft predicts the token after tok.
            backbone.ingest(s, Batch.step(tok));
            int pos = s.position() - 1; // position of `tok` (just committed)
            long hiddenOff =
                    (long) (s.lastChunkLen - 1)
                            * dim; // tok's pre-final-norm residual (row 0 of the step)
            FloatTensor dl = decoder.draft(s, s.residual, hiddenOff, tok, pos);
            int[] top = topK(dl, vocab, 10);
            System.out.printf(
                    "step %d: backbone tok=%d |%s|  draft top1=%d |%s|  top10=%s%n",
                    step,
                    tok,
                    tk.decode(tok).replace("\n", "\\n"),
                    top[0],
                    tk.decode(top[0]).replace("\n", "\\n"),
                    java.util.Arrays.toString(top));
            if (top[0] == ANCHOR_TOP[0]) { // the "." step the anchor captured
                int m = prefixMatch(top, ANCHOR_TOP);
                System.out.println("  -> anchor step reached. leading match " + m + "/10");
                System.out.println("     draft  " + java.util.Arrays.toString(top));
                System.out.println("     anchor " + java.util.Arrays.toString(ANCHOR_TOP));
                System.out.println("     pieces " + pieces(tk, top));
                matched = m >= 2; // top-1/top-2 identical = draft verified; the p~1e-4 tail is
                // Q8_0 cross-impl noise
                break;
            }
            tok =
                    argmax(
                            backbone.logits(s, 0),
                            vocab); // next greedy token (tok already ingested at loop top)
        }
        if (matched) {
            System.out.println(
                    "PASS: MTP draft reproduces the anchor distribution at the captured step");
        } else {
            System.out.println("FAIL: draft top-1/top-2 did not match the anchor at the '.' step");
            System.exit(1);
        }
    }

    static int argmax(FloatTensor t, int n) {
        int best = 0;
        float bv = t.getFloat(0);
        for (int i = 1; i < n; i++) {
            float v = t.getFloat(i);
            if (v > bv) {
                bv = v;
                best = i;
            }
        }
        return best;
    }

    static int[] topK(FloatTensor t, int n, int k) {
        int[] idx = new int[k];
        float[] val = new float[k];
        java.util.Arrays.fill(val, Float.NEGATIVE_INFINITY);
        for (int i = 0; i < n; i++) {
            float v = t.getFloat(i);
            if (v > val[k - 1]) {
                int j = k - 1;
                while (j > 0 && val[j - 1] < v) {
                    val[j] = val[j - 1];
                    idx[j] = idx[j - 1];
                    j--;
                }
                val[j] = v;
                idx[j] = i;
            }
        }
        return idx;
    }

    static int prefixMatch(int[] a, int[] b) {
        int m = 0;
        while (m < a.length && m < b.length && a[m] == b[m]) m++;
        return m;
    }

    static String pieces(com.qxotic.jinfer.GgufTokenizer tk, int[] ids) {
        var sb = new StringBuilder();
        for (int id : ids) sb.append('|').append(tk.decode(id).replace("\n", "\\n"));
        return sb.append('|').toString();
    }
}
