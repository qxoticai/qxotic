// Stage-3 gate for MTP speculative decode, in two layers:
//
//  HARD 1 (structural, any engine config): every emitted token equals the argmax of the verify row
//  that produced it (asserted via the TopRecorder) - the invariant that DEFINES speculative
//  correctness and is immune to cross-path numerics.
//
//  HARD 2 (exact, shape-invariant engine): under -Djinfer.disableJam=true (set below, before any
//  model class loads) the Java backends are bit-exact across chunk shapes (proven by
//  BatchVsStepProbe: batch rows == step rows, maxRel 0.0, including the rollback pattern), so MTP
//  greedy output must be TOKEN-IDENTICAL to plain greedy. Any mismatch here is a loop bug.
//
//  Under jam (production), batch-vs-step numerics are chunk-shape/alignment-dependent (documented:
//  BatchVsStepProbe shows argmax swaps in PLAIN chunked ingestion, no MTP involved - up to 1.6 rel
//  at flat positions on capped ISAs), so exact identity is not a valid gate there; the diagnostic
//  harness is MtpLockstepOracle (all jam-mode divergences must be top1/top2 swaps vs the oracle).
//
//   java ... com.qxotic.llm.Gemma4MtpIdentityTest
package com.qxotic.llm;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.CacheStore;
import com.qxotic.jinfer.cache.CachedSession;
import com.qxotic.jinfer.cache.PromptCache;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

public final class Gemma4MtpIdentityTest {

    static int failures;

    static final String[] PROMPTS = {
            "The capital of France is",
            "Write a short poem about the sea.",
            "def fibonacci(n):",
            "List the first ten prime numbers:",
            "Once upon a time, in a quiet village by the mountains,",
    };

    public static void main(String[] args) throws Exception {
        // shape-invariant engine BEFORE any model class initializes (JamMatMul reads it lazily,
        // but set it first to be independent of class-init order)
        System.setProperty("jinfer.disableJam", "true");

        Path model = Path.of("/home/mukel/Desktop/playground/models/unsloth/gemma-4-E2B-it-Q8_0.gguf");
        Path sidecar = Path.of("/home/mukel/Desktop/playground/models/unsloth/mtp-gemma-4-E2B-it.gguf");
        if (!Files.exists(model) || !Files.exists(sidecar)) {
            System.out.println("Gemma4MtpIdentityTest: model/sidecar absent, skipping");
            return;
        }
        Gemma4 m = Gemma4.loadModel(model, 4096, sidecar);
        var tk = m.tokenizer();
        int bos = tk.getSpecialTokens().getOrDefault("<bos>", 2);
        Set<Integer> stops = m.stopTokens();
        int maxTokens = 120;

        for (String prompt : PROMPTS) {
            int[] ids = withBos(bos, tk.encode(prompt));
            List<Integer> plain = plainGreedy(m, ids, maxTokens, stops);

            for (int depth : new int[]{1, 2}) {
                Gemma4.State ss = m.newState(4096, Math.max(16, ids.length));
                m.ingest(ss, Batch.prefill(ids));
                int[] rowViolations = {0};
                Gemma4Speculative.Result r = Gemma4Speculative.generate(m, ss, maxTokens, stops, depth,
                        (t, t1, l1, t2, l2) -> { if (t != t1) rowViolations[0]++; });   // HARD 1
                double acc = r.drafted() == 0 ? 0 : (double) r.accepted() / r.drafted();

                check(rowViolations[0] == 0, "d=" + depth + " verify-row invariant (0 violations)");
                boolean identical = plain.equals(r.tokens());                            // HARD 2
                check(identical, String.format(
                        "d=%d token-identical to plain greedy (%d tokens, %d forwards, accept %.0f%%): \"%s\"",
                        depth, r.tokens().size(), r.forwards(), 100 * acc, head(prompt)));
                if (!identical) diff(plain, r.tokens(), tk);

                check(ss.position() == ids.length + r.committed().size(),
                        "d=" + depth + " committed==KV (" + r.committed().size() + ")");
            }
        }

        // CachedSession integration: speculative decode on a session's state, then adopt(committed)
        {
            PromptCache<Gemma4.State> cache = new PromptCache<>(m.stateCodec().orElseThrow(), CacheStore.inMemory(),
                    1L << 30, PromptCache.modelSeed(model));
            CachedSession<Gemma4.State> session = CachedSession.resume(m, cache, m.newState(4096, 64), new long[0]);
            int[] ids = withBos(bos, tk.encode(PROMPTS[0]));
            session.ingest(List.of(Batch.prefill(ids)));
            Gemma4Speculative.Result r = Gemma4Speculative.generate(m, session.state(), 40, stops, 2);
            session.adopt(r.committed());
            check(session.length() == session.position(), "adopt: fingerprint stream in lockstep with KV ("
                    + session.length() + ")");
        }

        if (failures > 0) {
            System.out.println(failures + " failure(s)");
            System.exit(1);
        }
        System.out.println("Gemma4MtpIdentityTest: all identical - speculative greedy == plain greedy (shape-invariant engine)");
    }

    static List<Integer> plainGreedy(Gemma4 m, int[] ids, int maxTokens, Set<Integer> stops) {
        Gemma4.State s = m.newState(4096, Math.max(16, ids.length));
        m.ingest(s, Batch.prefill(ids));
        int vocab = m.config().vocabularySize();
        List<Integer> out = new ArrayList<>();
        int tok = m.logits(s, 0).argmax(0, vocab);
        while (out.size() < maxTokens && !stops.contains(tok)) {
            out.add(tok);
            m.ingest(s, Batch.step(tok));
            tok = m.logits(s, 0).argmax(0, vocab);
        }
        return out;
    }

    static int[] withBos(int bos, List<Integer> enc) {
        int[] ids = new int[enc.size() + 1];
        ids[0] = bos;
        for (int i = 0; i < enc.size(); i++) ids[i + 1] = enc.get(i);
        return ids;
    }

    static void diff(List<Integer> a, List<Integer> b, com.qxotic.jinfer.GgufTokenizer tk) {
        int i = 0;
        while (i < Math.min(a.size(), b.size()) && a.get(i).equals(b.get(i))) i++;
        System.out.println("  diverge at " + i + "/" + a.size() + " plain=" + (i < a.size() ? a.get(i) : -1)
                + " spec=" + (i < b.size() ? b.get(i) : -1));
        System.out.println("  plain: " + tk.decode(a.subList(0, Math.min(a.size(), i + 3))).replace("\n", "\\n"));
        System.out.println("  spec:  " + tk.decode(b.subList(0, Math.min(b.size(), i + 3))).replace("\n", "\\n"));
    }

    static String head(String s) {
        return s.length() <= 30 ? s : s.substring(0, 30) + "...";
    }

    static void check(boolean ok, String what) {
        if (ok) {
            System.out.println("ok:   " + what);
        } else {
            failures++;
            System.out.println("FAIL: " + what);
        }
    }
}
