package com.qxotic.jinfer.codecs;

import static org.junit.jupiter.api.Assertions.assertInstanceOf;

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
}
