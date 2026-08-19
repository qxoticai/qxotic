package com.qxotic.jinfer.codecs;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import org.junit.jupiter.api.Test;

class JavaSoundAudioDecoderTest {

    @Test
    void rejectsDecodedAudioPastTenMinutes() {
        IOException failure =
                assertThrows(
                        IOException.class,
                        () ->
                                JavaSoundAudioDecoder.readBounded(
                                        new ByteArrayInputStream(new byte[5]), 4));
        assertTrue(failure.getMessage().contains("4-byte limit"), failure.getMessage());
    }
}
