package com.qxotic.jinfer.codecs;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import org.junit.jupiter.api.Test;

class JavaSoundAudioDecoderTest {

    @Test
    void rejectsDecodedAudioPastTheLimit() {
        IOException failure =
                assertThrows(
                        Subprocess.OutputLimitExceeded.class,
                        () ->
                                JavaSoundAudioDecoder.readBounded(
                                        new ByteArrayInputStream(new byte[5]), 4));
        assertTrue(failure.getMessage().contains("4-byte limit"), failure.getMessage());
    }

    @Test
    void theMinuteCapIsValidatedAndItsCeilingFitsAFloatByteArray() {
        String property = FfmpegAudioDecoder.MAX_MINUTES;
        try {
            assertEquals(60, FfmpegAudioDecoder.maxMinutes());
            int ceiling = FfmpegAudioDecoder.MAX_MINUTES_CEILING;
            System.setProperty(property, Integer.toString(ceiling));
            assertEquals(ceiling, FfmpegAudioDecoder.maxMinutes());
            long bytes = 4L * FfmpegAudioDecoder.maxSamples(ceiling);
            assertTrue(bytes < Integer.MAX_VALUE - 8, "ceiling overflows: " + bytes);
            assertEquals(bytes, 4 * FfmpegAudioDecoder.maxSamples(ceiling), "int math wraps");
            for (String bad : new String[] {"0", "-1", "an hour", Integer.toString(ceiling + 1)}) {
                System.setProperty(property, bad);
                var e =
                        assertThrows(
                                IllegalArgumentException.class, FfmpegAudioDecoder::maxMinutes);
                assertTrue(e.getMessage().startsWith(property), e.getMessage());
            }
        } finally {
            System.clearProperty(property);
        }
    }
}
