package com.qxotic.jinfer.codecs;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

/** The media backends a JVM selects, and the reflective class names behind the JDK ones. */
class JvmBackendSelectionTest {

    /** ffmpeg wherever it runs, so a JVM reads every format a native image does. */
    @Test
    void theDefaultIsFfmpegElseTheJdk() {
        Assumptions.assumeTrue(System.getProperty("jinfer.imageDecoder") == null);
        Assumptions.assumeTrue(System.getProperty("jinfer.audioDecoder") == null);
        boolean ffmpeg = Codecs.ffmpegOnPath();
        assertEquals(
                ffmpeg ? FfmpegImageDecoder.class : ImageIoDecoder.class,
                ImageCodec.select().getClass());
        assertEquals(
                ffmpeg ? FfmpegAudioDecoder.class : JavaSoundAudioDecoder.class,
                AudioCodec.select().getClass());
    }

    /** Selected by name, the JDK backends load: their class names survive package renames. */
    @Test
    void theJdkBackendsLoadWhenSelected() {
        Assumptions.assumeTrue(System.getProperty("jinfer.imageDecoder") == null);
        Assumptions.assumeTrue(System.getProperty("jinfer.audioDecoder") == null);
        System.setProperty("jinfer.imageDecoder", "imageio");
        System.setProperty("jinfer.audioDecoder", "javasound");
        try {
            assertInstanceOf(ImageIoDecoder.class, ImageCodec.select());
            assertInstanceOf(JavaSoundAudioDecoder.class, AudioCodec.select());
        } finally {
            System.clearProperty("jinfer.imageDecoder");
            System.clearProperty("jinfer.audioDecoder");
        }
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
