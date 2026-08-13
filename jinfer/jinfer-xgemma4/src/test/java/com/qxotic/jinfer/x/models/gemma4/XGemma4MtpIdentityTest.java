// Stage-3 gate for MTP speculative decode, x tree, in two layers:
//
//  HARD 1 (structural, any engine config): every emitted token equals the argmax of the verify
//  row that produced it (asserted via the SpeculationAudit tap) - the invariant that DEFINES
//  speculative correctness and is immune to cross-path numerics.
//
//  HARD 2 (exact, shape-invariant engine): under -Djinfer.disableJam=true (set below, before any
//  model class loads) the Java backends are bit-exact across chunk shapes, so MTP greedy output
//  must be TOKEN-IDENTICAL to plain greedy. Any mismatch here is a loop bug.
//
// Plus: committed==KV accounting on every pass, and the CachedSession.adopt(committed) lockstep.
package com.qxotic.jinfer.x.models.gemma4;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.testkit.TestModels;
import com.qxotic.jinfer.x.Views;
import com.qxotic.jinfer.x.boundary.Batch;
import com.qxotic.jinfer.x.boundary.ContentKey;
import com.qxotic.jinfer.x.cache.BlockTree;
import com.qxotic.jinfer.x.cache.CacheStore;
import com.qxotic.jinfer.x.cache.CachedSession;
import com.qxotic.jinfer.x.kernels.Ops;
import com.qxotic.jinfer.x.llm.Generator.Constraints;
import com.qxotic.jinfer.x.llm.Sampler;
import com.qxotic.jinfer.x.llm.SpecialTokens;
import com.qxotic.jinfer.x.llm.SpeculativeDecoding.SpeculationResult;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import java.lang.foreign.Arena;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

class XGemma4MtpIdentityTest {

    private static final String MODEL_REF = "hf.co/unsloth/gemma-4-E2B-it-GGUF:Q8_0";
    private static final String SIDECAR_REF =
            "hf.co/unsloth/gemma-4-E2B-it-GGUF/mtp-gemma-4-E2B-it.gguf";

    static final String[] PROMPTS = {
        "The capital of France is",
        "Write a short poem about the sea.",
        "def fibonacci(n):",
        "List the first ten prime numbers:",
        "Once upon a time, in a quiet village by the mountains,",
    };

    @Test
    @Tag("driver")
    void speculativeGreedyIsPlainGreedy() throws Exception {
        Path model = TestModels.require(MODEL_REF);
        Path sidecar = TestModels.require(SIDECAR_REF);
        // shape-invariant engine BEFORE any model class initializes (the backends read it lazily,
        // but set it first to be independent of class-init order)
        System.setProperty("jinfer.disableJam", "true");

        Gemma4 m = Gemma4.loadWithMtp(model, sidecar, Arena.ofAuto());
        assertTrue(m.speculationReady(), "sidecar attached");
        Tokenizer tk = m.tokenizer();
        int bos = SpecialTokens.find(tk, "<bos>").orElse(2);
        Set<Integer> stops = stopTokens(tk);
        int maxTokens = 120;

        for (String prompt : PROMPTS) {
            int[] ids = withBos(bos, tk.encode(prompt).toList());
            List<Integer> plain = plainGreedy(m, ids, maxTokens, stops);

            for (int depth : new int[] {1, 2}) {
                try (Gemma4.State ss = m.newState(4096, Math.max(16, ids.length))) {
                    m.ingest(ss, Batch.prefill(ids));
                    int[] violations = {0};
                    SpeculationResult r =
                            m.speculate(
                                    ss,
                                    Sampler.ARGMAX,
                                    new Constraints(maxTokens, Duration.ZERO, stops),
                                    depth,
                                    null,
                                    (token, targetArgmax) -> {
                                        if (token != targetArgmax) violations[0]++;
                                    });
                    double acc = r.drafted() == 0 ? 0 : (double) r.accepted() / r.drafted();

                    // HARD 1
                    assertEquals(
                            0,
                            violations[0],
                            "d=" + depth + " verify-row invariant violations: " + prompt);
                    // HARD 2
                    List<Integer> emitted = toList(r.emitted());
                    assertEquals(
                            plain,
                            emitted,
                            String.format(
                                    "d=%d not token-identical to plain greedy (%d tokens, %d"
                                            + " forwards, accept %.0f%%): \"%s\"",
                                    depth, emitted.size(), r.forwards(), 100 * acc, prompt));
                    // KV accounting
                    assertEquals(
                            ids.length + r.committed().length(),
                            ss.position(),
                            "d=" + depth + " committed==KV: " + prompt);
                }
            }
        }

        // CachedSession integration: speculative decode on a session's state, then
        // adopt(committed) - the fingerprint stream must stay in lockstep with the KV.
        {
            BlockTree<Gemma4.State> cache =
                    new BlockTree<>(
                            m.stateCodec().orElseThrow(),
                            CacheStore.inMemory(),
                            1L << 30,
                            ContentKey.sha256(model.toString().getBytes(StandardCharsets.UTF_8)));
            CachedSession<Gemma4.State> session =
                    CachedSession.start(m, cache, m.newState(4096, 64));
            int[] ids = withBos(bos, tk.encode(PROMPTS[0]).toList());
            session.ingest(List.of(Batch.prefill(ids)));
            SpeculationResult r =
                    m.speculate(
                            session.state(),
                            Sampler.ARGMAX,
                            new Constraints(40, Duration.ZERO, stops),
                            2,
                            null,
                            null);
            session.adopt(r.committed());
            assertEquals(
                    session.length(),
                    session.position(),
                    "adopt: fingerprint stream in lockstep with KV");
        }
    }

    static Set<Integer> stopTokens(Tokenizer tk) {
        Set<Integer> stops = new LinkedHashSet<>();
        for (String name : new String[] {"<end_of_turn>", "<eos>"}) {
            if (tk.vocabulary().contains(name)) {
                stops.add(tk.vocabulary().id(name));
            }
        }
        return stops;
    }

    static int argmaxLastRow(Gemma4 m, Gemma4.State s) {
        int vocab = m.config().vocabularySize();
        return Ops.argmax(Views.castToSegmentBacked(m.logits(s), "logits"), 0, vocab);
    }

    static List<Integer> plainGreedy(Gemma4 m, int[] ids, int maxTokens, Set<Integer> stops) {
        try (Gemma4.State s = m.newState(4096, Math.max(16, ids.length))) {
            m.ingest(s, Batch.prefill(ids));
            List<Integer> out = new ArrayList<>();
            int tok = argmaxLastRow(m, s);
            while (out.size() < maxTokens && !stops.contains(tok)) {
                out.add(tok);
                m.ingest(s, Batch.step(tok));
                tok = argmaxLastRow(m, s);
            }
            return out;
        }
    }

    static int[] withBos(int bos, List<Integer> enc) {
        int[] ids = new int[enc.size() + 1];
        ids[0] = bos;
        for (int i = 0; i < enc.size(); i++) ids[i + 1] = enc.get(i);
        return ids;
    }

    static List<Integer> toList(IntSequence seq) {
        List<Integer> out = new ArrayList<>(seq.length());
        for (int i = 0; i < seq.length(); i++) out.add(seq.intAt(i));
        return out;
    }
}
