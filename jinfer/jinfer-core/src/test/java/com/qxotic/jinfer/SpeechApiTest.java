package com.qxotic.jinfer;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.media.Media;
import java.util.List;
import org.junit.jupiter.api.Test;

/** The speech contract that holds without any weights: joining clips, and the options floor. */
final class SpeechApiTest {

    @Test
    void clipsJoinEndToEndAtOneRate() {
        Media.Audio joined =
                Media.Audio.concat(
                        List.of(
                                new Media.Audio(new float[] {0.1f, 0.2f}, 24000, 1),
                                new Media.Audio(new float[] {0.3f}, 24000, 1)));

        assertArrayEquals(new float[] {0.1f, 0.2f, 0.3f}, joined.pcm());
        assertEquals(24000, joined.sampleRate());
        assertEquals(1, joined.channels());
    }

    @Test
    void aRateMismatchIsRefusedRatherThanResampledByAccident() {
        // silently picking one rate would play the other clip back at the wrong pitch
        IllegalArgumentException e =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                Media.Audio.concat(
                                        List.of(
                                                new Media.Audio(new float[] {0f}, 24000, 1),
                                                new Media.Audio(new float[] {0f}, 22050, 1))));
        assertTrue(e.getMessage().contains("24000"), e.getMessage());

        assertThrows(
                IllegalArgumentException.class,
                () ->
                        Media.Audio.concat(
                                List.of(
                                        new Media.Audio(new float[] {0f}, 24000, 1),
                                        new Media.Audio(new float[] {0f, 0f}, 24000, 2))));
    }

    @Test
    void anEmptyListIsNotAWaveform() {
        assertThrows(IllegalArgumentException.class, () -> Media.Audio.concat(List.of()));
    }

    @Test
    void noneMeansTheModelsOwnDefaults() {
        assertNull(SpeechOptions.NONE.speed(), "NONE overrides nothing");
        assertEquals(1.25, SpeechOptions.speed(1.25).speed());
    }

    @Test
    void aRateThatCannotMultiplyDurationsIsRefused() {
        assertThrows(IllegalArgumentException.class, () -> SpeechOptions.speed(0));
        assertThrows(IllegalArgumentException.class, () -> SpeechOptions.speed(-1.5));
        assertThrows(IllegalArgumentException.class, () -> SpeechOptions.speed(Double.NaN));
        assertThrows(
                IllegalArgumentException.class,
                () -> SpeechOptions.speed(Double.POSITIVE_INFINITY));
    }
}
