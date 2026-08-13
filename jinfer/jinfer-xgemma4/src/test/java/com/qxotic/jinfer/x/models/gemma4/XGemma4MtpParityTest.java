// Old-vs-x MTP parity gate, floor engine (-Djinfer.disableJam=true, set before any model class
// loads): the same checkpoint and sidecar decode the same prompt in BOTH trees, and the whole
// speculative pass must agree - tokens AND the draft-quality counters (drafted/accepted/forwards).
// Token identity alone cannot pin the draft decoder (a subtly wrong draft merely wastes forwards
// on rejection); the acceptance counters can.
package com.qxotic.jinfer.x.models.gemma4;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jinfer.testkit.TestModels;
import com.qxotic.jinfer.x.boundary.Batch;
import com.qxotic.jinfer.x.llm.Generator.Constraints;
import com.qxotic.jinfer.x.llm.Sampler;
import com.qxotic.jinfer.x.llm.SpecialTokens;
import com.qxotic.jinfer.x.llm.SpeculativeDecoding.SpeculationResult;
import com.qxotic.toknroll.Tokenizer;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.time.Duration;
import java.util.Set;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

class XGemma4MtpParityTest {

    private static final String MODEL_REF = "hf.co/unsloth/gemma-4-E2B-it-GGUF:Q8_0";
    private static final String SIDECAR_REF =
            "hf.co/unsloth/gemma-4-E2B-it-GGUF/mtp-gemma-4-E2B-it.gguf";

    @Test
    @Tag("driver")
    void oldAndXSpeculateIdentically() throws Exception {
        Path model = TestModels.require(MODEL_REF);
        Path sidecar = TestModels.require(SIDECAR_REF);
        // shape-invariant engine BEFORE any model class initializes
        System.setProperty("jinfer.disableJam", "true");

        var old =
                com.qxotic.jinfer.models.gemma4.Gemma4.loadWithMtp(model, sidecar, Arena.ofAuto());
        Gemma4 x = Gemma4.loadWithMtp(model, sidecar, Arena.ofAuto());
        Tokenizer tk = x.tokenizer();
        int bos = SpecialTokens.find(tk, "<bos>").orElse(2);
        Set<Integer> stops = XGemma4MtpIdentityTest.stopTokens(tk);
        int maxTokens = 120;

        for (String prompt : XGemma4MtpIdentityTest.PROMPTS) {
            int[] ids = XGemma4MtpIdentityTest.withBos(bos, tk.encode(prompt).toList());
            for (int depth : new int[] {1, 2}) {
                var oldState = old.newState(4096, Math.max(16, ids.length));
                old.ingest(oldState, com.qxotic.jinfer.Batch.prefill(ids));
                var oldResult =
                        com.qxotic.jinfer.models.gemma4.Gemma4Speculative.generate(
                                old, oldState, maxTokens, stops, depth);

                SpeculationResult xResult;
                try (Gemma4.State xState = x.newState(4096, Math.max(16, ids.length))) {
                    x.ingest(xState, Batch.prefill(ids));
                    xResult =
                            x.speculate(
                                    xState,
                                    Sampler.ARGMAX,
                                    new Constraints(maxTokens, Duration.ZERO, stops),
                                    depth,
                                    null,
                                    null);
                }

                assertEquals(
                        oldResult.tokens(),
                        XGemma4MtpIdentityTest.toList(xResult.emitted()),
                        "d=" + depth + " emitted tokens diverge: " + prompt);
                assertEquals(
                        oldResult.committed(),
                        XGemma4MtpIdentityTest.toList(xResult.committed()),
                        "d=" + depth + " committed tokens diverge: " + prompt);
                assertEquals(
                        oldResult.drafted(),
                        xResult.drafted(),
                        "d=" + depth + " drafted diverges: " + prompt);
                assertEquals(
                        oldResult.accepted(),
                        xResult.accepted(),
                        "d=" + depth + " accepted diverges (draft decoder port): " + prompt);
                assertEquals(
                        oldResult.forwards(),
                        xResult.forwards(),
                        "d=" + depth + " forwards diverges: " + prompt);
            }
        }
    }
}
