package com.qxotic.jinfer.x.cli;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.testkit.TestModels;
import com.qxotic.jinfer.x.Views;
import com.qxotic.jinfer.x.boundary.Batch;
import com.qxotic.jinfer.x.boundary.ContextState;
import com.qxotic.jinfer.x.chat.LoadedModel;
import com.qxotic.jinfer.x.chat.Models;
import com.qxotic.jinfer.x.llm.Generator;
import com.qxotic.jinfer.x.llm.Generator.Constraints;
import com.qxotic.jinfer.x.llm.Sampler;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.time.Duration;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/** X-only state recycling and prefill gates at the public model-loading boundary. */
@Tag("integration")
class ModelStateContractIT {

    private static final String GEMMA4_MOE =
            "hf.co/unsloth/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-Q8_0.gguf";
    private static final String QWEN35 = "hf.co/unsloth/Qwen3.5-2B-GGUF:Q8_0";

    @ParameterizedTest(name = "reset identity: {0}")
    @ValueSource(
            strings = {
                "hf.co/unsloth/gemma-4-E2B-it-qat-GGUF/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf",
                "hf.co/unsloth/gpt-oss-20b-GGUF:Q8_0",
                "hf.co/ibm-granite/granite-4.1-3b-GGUF:Q8_0",
                "hf.co/LiquidAI/LFM2.5-350M-GGUF:Q8_0",
                "hf.co/unsloth/Llama-3.2-1B-Instruct-GGUF:Q8_0",
                "hf.co/bartowski/nvidia_Nemotron-Cascade-2-30B-A3B-GGUF:Q8_0",
                QWEN35
            })
    void resetMatchesAFreshState(String ref) throws Exception {
        try (Arena weights = Arena.ofShared()) {
            assertResetIdentity(Models.load(TestModels.require(ref), weights));
        }
    }

    @Test
    void repeatedGemmaMoePrefillsStayStable() throws Exception {
        try (Arena weights = Arena.ofShared()) {
            assertRepeatedPrefills(Models.load(TestModels.require(GEMMA4_MOE), weights));
        }
    }

    @Test
    void qwen35BatchedAndSteppedPrefillChooseTheSameNextToken() throws Exception {
        try (Arena weights = Arena.ofShared()) {
            assertBatchedAndSteppedPrefill(Models.load(TestModels.require(QWEN35), weights));
        }
    }

    private static <S extends ContextState> void assertResetIdentity(LoadedModel<S> loaded) {
        int[] dirtyingPrompt = loaded.tokenizer().encode("The capital of France is").toArray();
        int[] checkedPrompt = loaded.tokenizer().encode("Once upon a time there was").toArray();
        try (S recycled = loaded.model().newState(1024, 64);
                S fresh = loaded.model().newState(1024, 64)) {
            for (int i = 0; i < 4; i++) {
                generate(loaded, recycled, dirtyingPrompt);
                recycled.reset();
            }
            assertArrayEquals(
                    generate(loaded, fresh, checkedPrompt),
                    generate(loaded, recycled, checkedPrompt),
                    "reset left model history behind");
        }
    }

    private static <S extends ContextState> int[] generate(
            LoadedModel<S> loaded, S state, int[] prompt) {
        return Generator.generate(
                        loaded.model(),
                        state,
                        prompt,
                        Sampler.ARGMAX,
                        new Constraints(24, Duration.ZERO, loaded.stopTokens()),
                        token -> true)
                .tokens();
    }

    private static <S extends ContextState> void assertRepeatedPrefills(LoadedModel<S> loaded) {
        int[] prompt = loaded.tokenizer().encode("The capital of France is").toArray();
        float[] reference = null;
        for (int repetition = 0; repetition < 8; repetition++) {
            float[] logits = prefill(loaded, prompt, 512);
            if (repetition < 4) continue;
            if (reference == null) reference = logits;
            else assertTrue(maxDifference(reference, logits) <= 1e-2, "prefill drifted");
        }
    }

    private static <S extends ContextState> void assertBatchedAndSteppedPrefill(
            LoadedModel<S> loaded) {
        int[] prompt =
                loaded.tokenizer()
                        .encode(
                                "The expedition logged river depth, canopy density and soil"
                                        + " acidity at every station; readings were nominal and the"
                                        + " weather held clear. Summarize the day in one sentence.")
                        .toArray();
        float[] batched = prefill(loaded, prompt, 512);
        float[] stepped;
        try (S state = loaded.model().newState(4096, 16)) {
            for (int token : prompt) loaded.model().ingest(state, Batch.step(token));
            stepped = snapshot(loaded.model().logits(state));
        }
        assertEquals(argmax(batched), argmax(stepped));
    }

    private static <S extends ContextState> float[] prefill(
            LoadedModel<S> loaded, int[] prompt, int batchCapacity) {
        try (S state = loaded.model().newState(4096, batchCapacity)) {
            loaded.model().ingest(state, Batch.prefill(prompt));
            return snapshot(loaded.model().logits(state));
        }
    }

    private static float[] snapshot(MemoryView<?> view) {
        var logits = Views.castToSegmentBacked(view, "logits");
        float[] values = new float[Math.toIntExact(logits.shape().size())];
        for (int i = 0; i < values.length; i++) values[i] = Views.getFloat(logits, i, "logits");
        return values;
    }

    private static double maxDifference(float[] a, float[] b) {
        double max = 0;
        for (int i = 0; i < a.length; i++) max = Math.max(max, Math.abs(a[i] - b[i]));
        return max;
    }

    private static int argmax(float[] values) {
        int best = 0;
        for (int i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
        return best;
    }
}
