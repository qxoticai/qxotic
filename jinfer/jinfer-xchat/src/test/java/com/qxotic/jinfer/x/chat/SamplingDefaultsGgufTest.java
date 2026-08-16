package com.qxotic.jinfer.x.chat;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

import com.qxotic.format.gguf.Builder;
import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.x.chat.LoadedModel.SamplingDefaults;
import com.qxotic.jinfer.x.llm.Sampling;
import org.junit.jupiter.api.Test;

/**
 * Tests the {@code general.sampling.*} GGUF metadata layer of {@link SamplingDefaults}, in
 * particular llama.cpp's {@code top_k <= 0} sentinel convention.
 */
class SamplingDefaultsGgufTest {

    private static GGUF ggufWithTopK(int topK) {
        return Builder.newBuilder().putInteger("general.sampling.top_k", topK).build();
    }

    @Test
    void absentTopKIsNoOpinion() {
        SamplingDefaults defaults = SamplingDefaults.fromGGUF(Builder.newBuilder().build());

        assertNull(defaults.topK());
        assertEquals(40, defaults.effectiveTopK());
        assertEquals(40, defaults.resolve(null, null, null, null, null).topK());
    }

    @Test
    void positiveTopKIsCarriedVerbatim() {
        SamplingDefaults defaults = SamplingDefaults.fromGGUF(ggufWithTopK(64));

        assertEquals(64, defaults.topK());
        assertEquals(64, defaults.resolve(null, null, null, null, null).topK());
    }

    @Test
    void minusOneTopKMeansExplicitlyDisabled() {
        SamplingDefaults defaults = SamplingDefaults.fromGGUF(ggufWithTopK(-1));

        assertEquals(0, defaults.topK());
        // Regression: resolving the -1 sentinel used to throw IAE from Sampling's constructor.
        Sampling sampling =
                assertDoesNotThrow(() -> defaults.resolve(null, null, null, null, null));
        assertEquals(0, sampling.topK());
    }

    @Test
    void zeroTopKMeansExplicitlyDisabled() {
        assertEquals(0, SamplingDefaults.fromGGUF(ggufWithTopK(0)).topK());
    }

    @Test
    void garbageNegativeTopKIsClampedToDisabled() {
        assertEquals(0, SamplingDefaults.fromGGUF(ggufWithTopK(Integer.MIN_VALUE)).topK());
    }

    @Test
    void disabledTopKShadowsLowerLayers() {
        // An explicit container statement wins over the port's recommendation and the baseline.
        SamplingDefaults port = new SamplingDefaults(null, null, 20, null);
        SamplingDefaults fromGguf = SamplingDefaults.fromGGUF(ggufWithTopK(-1)).withFallback(port);

        assertEquals(0, fromGguf.resolve(null, null, null, null, null).topK());
    }

    @Test
    void otherSamplingKeysPassThroughUntouched() {
        GGUF gguf =
                Builder.newBuilder()
                        .putFloat("general.sampling.temp", 1.0f)
                        .putFloat("general.sampling.top_p", 0.95f)
                        .putFloat("general.sampling.min_p", 0.05f)
                        .build();
        SamplingDefaults defaults = SamplingDefaults.fromGGUF(gguf);

        assertEquals(1.0f, defaults.temperature());
        assertEquals(0.95f, defaults.topP());
        assertEquals(0.05f, defaults.minP());
    }
}
