package com.qxotic.jinfer.models.mellum;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.cache.PromptCache;
import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.testkit.TestModels;
import com.qxotic.jota.DataType;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("integration")
final class MellumIntegrationTest {
    private static final int PARIS = 12330; // " Paris"

    @Test
    void loadsThroughTheProviderWithTheCardsSettings() throws Exception {
        try (Arena arena = Arena.ofShared()) {
            LoadedModel<?> loaded = Models.load(model(), arena);
            Mellum model = assertInstanceOf(Mellum.class, loaded.model());
            assertInstanceOf(MellumChatTemplate.class, loaded.template().orElseThrow());
            assertEquals(0.6f, loaded.samplingDefaults().temperature());
            assertEquals(0.95f, loaded.samplingDefaults().topP());
            assertEquals(20, loaded.samplingDefaults().topK());
            assertEquals(28, loaded.stopTokens().iterator().next(), "<|im_end|> ends the turn");

            Mellum.Configuration config = model.configuration();
            assertEquals(28, config.numberOfLayers());
            assertEquals(2_304, config.embeddingLength());
            assertEquals(98_304, config.vocabularySize());
            assertEquals(131_072, config.maxContextLength());
            assertEquals(32, config.numberOfHeads());
            assertEquals(4, config.numberOfKeyValueHeads());
            assertEquals(128, config.headSize());
            assertEquals(64, config.expertCount());
            assertEquals(8, config.expertUsedCount());
            assertEquals(896, config.expertFeedForwardLength());
            assertEquals(1_024, config.slidingWindow());
            for (int layer = 0; layer < 28; layer++)
                assertEquals(layer % 4 != 3, config.isSwa()[layer], "layer " + layer);
            assertEquals(16f, config.ropeScalingFactor());
            assertEquals(8_192, config.ropeOriginalContext());

            int[] prompt = model.tokenizer().encodeToArray("The capital of France is");
            assertArrayEquals(new int[] {629, 5782, 332, 8438, 359}, prompt);
            try (Mellum.State state = model.newState(64, prompt.length)) {
                model.ingest(state, Batch.prefill(prompt));
                assertEquals(PARIS, argmax(logits(model, state)));
            }
        }
    }

    @Test
    void batchedAndIncrementalInferenceAgreeAndResetReplays() throws Exception {
        try (Arena arena = Arena.ofShared()) {
            Mellum model = Mellum.loadModel(model(), arena);
            int[] tokens = model.tokenizer().encodeToArray("The quick brown fox jumps over the");

            float[] batched;
            try (Mellum.State state = model.newState(64, tokens.length)) {
                assertEquals(DataType.FP16, state.keyCache[0].dataType());
                assertEquals(DataType.FP16, state.valueCache[3].dataType());
                assertEquals(DataType.FP32, state.batchK.dataType());
                model.ingest(state, Batch.prefill(tokens));
                batched = logits(model, state);
            }

            try (Mellum.State state = model.newState(64, 1)) {
                for (int token : tokens) model.ingest(state, Batch.step(token));
                float[] incremental = logits(model, state);
                double drift = rmse(batched, incremental);
                System.out.printf(
                        "Mellum batch/step parity: rmse=%.6f max=%.6f cosine=%.9f%n",
                        drift,
                        maxAbsDifference(batched, incremental),
                        cosineSimilarity(batched, incremental));
                assertEquals(argmax(batched), argmax(incremental));
                // decode requantizes activations for the int8 gemv where prefill runs the gemm,
                // and a near-tied route in the 64-expert top-8 router turns that rounding into a
                // visible drift (the Java floor agrees to 1e-2; the shape of the distribution is
                // what has to hold)
                assertTrue(cosineSimilarity(batched, incremental) > .995, "cosine " + drift);

                state.reset();
                for (int token : tokens) model.ingest(state, Batch.step(token));
                // a replay is the same arithmetic again, up to the JIT's warm-up rounding
                assertArrayEquals(incremental, logits(model, state), .01f);
            }
        }
    }

    /** A prompt longer than the window: every sliding layer's ring wraps, and restores intact. */
    @Test
    void promptCacheRestoresFullRowsAndSlidingWindowRingsPastTheWindow() throws Exception {
        Path path = model();
        try (Arena arena = Arena.ofShared()) {
            Mellum model = Mellum.loadModel(path, arena);
            int[] tokens = new Random(7).ints(1_200, 1_000, 90_000).toArray();
            // three prefill blocks: the third one wraps every ring (positions 1024+ overwrite
            // slots 0+), so the capture at its end and the restore below both cross the seam
            List<Batch> prompt =
                    List.of(
                            Batch.prefill(Arrays.copyOfRange(tokens, 0, 400)),
                            Batch.prefill(Arrays.copyOfRange(tokens, 400, 800)),
                            Batch.prefill(Arrays.copyOfRange(tokens, 800, tokens.length - 1)),
                            Batch.step(tokens[tokens.length - 1]));
            PromptCache.Options options =
                    PromptCache.Options.DEFAULTS
                            .withRetainedSessions(0)
                            .withContextCapacity(2_048)
                            .withBlockBudget(256L << 20);
            try (PromptCache<Mellum.State> cache =
                    PromptCache.of(model, Models.modelSeed(path), options)) {
                CacheResult fresh = cachedLogits(cache, model, prompt);
                CacheResult restored = cachedLogits(cache, model, prompt);

                assertEquals(PromptCache.Tier.FRESH, fresh.tier());
                assertEquals(PromptCache.Tier.BLOCKS, restored.tier());
                assertEquals(tokens.length - 1, restored.restored());
                assertArrayEquals(fresh.logits(), restored.logits(), 1e-4f);
            }
        }
    }

    private static Path model() {
        return TestModels.require(MellumChatTemplateTest.MODEL_REF);
    }

    private static float[] logits(Mellum model, Mellum.State state) {
        return Views.toFloatArray(
                Views.castToSegmentBacked(model.logits(state), "logits"), "logits");
    }

    private static CacheResult cachedLogits(
            PromptCache<Mellum.State> cache, Mellum model, List<Batch> prompt) {
        return cache.serve(
                prompt,
                (state, serving) ->
                        new CacheResult(serving.tier(), serving.restored(), logits(model, state)));
    }

    private static int argmax(float[] values) {
        int best = 0;
        for (int i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
        return best;
    }

    private static float maxAbsDifference(float[] left, float[] right) {
        float max = 0f;
        for (int i = 0; i < left.length; i++) max = Math.max(max, Math.abs(left[i] - right[i]));
        return max;
    }

    private static double rmse(float[] left, float[] right) {
        double squareSum = 0;
        for (int i = 0; i < left.length; i++) {
            double difference = left[i] - right[i];
            squareSum += difference * difference;
        }
        return Math.sqrt(squareSum / left.length);
    }

    private static double cosineSimilarity(float[] left, float[] right) {
        double dot = 0, leftSquareSum = 0, rightSquareSum = 0;
        for (int i = 0; i < left.length; i++) {
            dot += (double) left[i] * right[i];
            leftSquareSum += (double) left[i] * left[i];
            rightSquareSum += (double) right[i] * right[i];
        }
        return dot / Math.sqrt(leftSquareSum * rightSquareSum);
    }

    private record CacheResult(PromptCache.Tier tier, int restored, float[] logits) {}
}
