package com.qxotic.jinfer.chat;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

import org.junit.jupiter.api.Test;

/** Pins the precedence chain from container metadata through port defaults to engine defaults. */
class SamplingDefaultsTest {

    @Test
    void containerValuesWinFieldByFieldOverPortFallback() {
        var container = new LoadedModel.SamplingDefaults(0.2f, null, 80, null);
        var port = new LoadedModel.SamplingDefaults(1.0f, 0.9f, null, 0.1f);
        var merged = container.withFallback(port);
        assertEquals(0.2f, merged.temperature());
        assertEquals(0.9f, merged.topP());
        assertEquals(80, merged.topK());
        assertEquals(0.1f, merged.minP());
    }

    @Test
    void absentRecommendationsFallThroughToTheEngineBaseline() {
        var defaults =
                LoadedModel.SamplingDefaults.NONE.withFallback(LoadedModel.SamplingDefaults.NONE);
        assertNull(defaults.temperature());
        assertEquals(0.8f, defaults.effectiveTemperature());
        assertEquals(0.95f, defaults.effectiveTopP());
        assertEquals(40, defaults.effectiveTopK());
        assertEquals(0.05f, defaults.effectiveMinP());
    }

    @Test
    void explicitRequestValuesWinOverEveryDefault() {
        var defaults = new LoadedModel.SamplingDefaults(1.0f, 0.9f, 64, 0.1f);
        var sampling = defaults.resolve(0.3f, null, 12, null, 7L);
        assertEquals(0.3f, sampling.temperature());
        assertEquals(0.9f, sampling.topP());
        assertEquals(12, sampling.topK());
        assertEquals(0.1f, sampling.minP());
        assertEquals(7L, sampling.seed());
    }
}
