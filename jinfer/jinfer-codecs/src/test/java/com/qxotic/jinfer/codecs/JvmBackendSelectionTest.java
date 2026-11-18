package com.qxotic.jinfer.codecs;

import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

/** Guards the reflective class names used by the JVM-default media backends. */
class JvmBackendSelectionTest {

    @Test
    void defaultBackendsSurvivePackageRenames() {
        Assumptions.assumeTrue(System.getProperty("jinfer.imageDecoder") == null);
        Assumptions.assumeTrue(System.getProperty("jinfer.audioDecoder") == null);
        Assumptions.assumeFalse(Codecs.nativeImage());

        assertInstanceOf(ImageIoDecoder.class, ImageCodec.decoder());
        assertInstanceOf(JavaSoundAudioDecoder.class, AudioCodec.decoder());
    }

    /** A backend this runtime lacks is refused naming the switch, never swapped for ffmpeg. */
    @Test
    void anUnavailableBackendIsRefusedNotReplaced() {
        IllegalStateException refused =
                assertThrows(
                        IllegalStateException.class,
                        () ->
                                Codecs.reflect(
                                        "com.qxotic.jinfer.codecs.Missing", "jinfer.imageDecoder"));
        assertTrue(
                refused.getMessage().endsWith("-Djinfer.imageDecoder=ffmpeg decodes without it"),
                refused.getMessage());
    }
}
