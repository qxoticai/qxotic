package com.qxotic.jinfer.chat;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

import org.junit.jupiter.api.Test;

class SamplingDefaultsTest {

    @Test
    void containerValuesWinOverPortFallback() {
        var container = new LoadedModel.SamplingDefaults(0.2f, null);
        var port = new LoadedModel.SamplingDefaults(1.0f, 0.9f);
        var merged = container.withFallback(port);
        assertEquals(0.2f, merged.temperature());
        assertEquals(0.9f, merged.topP()); // absent in the container -> the port's value
    }

    @Test
    void noneFallsThroughToEngineBaseline() {
        var merged =
                LoadedModel.SamplingDefaults.NONE.withFallback(LoadedModel.SamplingDefaults.NONE);
        assertNull(merged.temperature());
        assertEquals(0.8f, merged.effectiveTemperature());
        assertEquals(0.95f, merged.effectiveTopP());
    }

    @Test
    void portRecommendationBeatsBaseline() {
        var merged =
                LoadedModel.SamplingDefaults.NONE.withFallback(
                        new LoadedModel.SamplingDefaults(1.0f, 1.0f));
        assertEquals(1.0f, merged.effectiveTemperature());
        assertEquals(1.0f, merged.effectiveTopP());
    }
}
