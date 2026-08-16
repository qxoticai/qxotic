package com.qxotic.jinfer.models.gemma4;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.format.gguf.Builder;
import com.qxotic.format.gguf.GGUF;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;

/** A media sidecar must be rejected from its header before any incompatible tensor is mapped. */
class Gemma4MediaPairingTest {

    private static final Path SIDECAR = Path.of("mmproj-F32.gguf");

    @Test
    void matchingProjectorPasses() {
        assertDoesNotThrow(() -> Gemma4.validateMediaPairing(SIDECAR, vision(1536), 1536));
    }

    @Test
    void projectorFromAnotherModelSizeNamesBothWidthsAndTheRemedy() {
        IllegalArgumentException failure =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> Gemma4.validateMediaPairing(SIDECAR, vision(1536), 3840));
        assertTrue(failure.getMessage().contains("1536"), failure.getMessage());
        assertTrue(failure.getMessage().contains("3840"), failure.getMessage());
        assertTrue(failure.getMessage().contains("same Gemma 4 size"), failure.getMessage());
    }

    @Test
    void fileWithoutAProjectorIsRejected() {
        IllegalArgumentException failure =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                Gemma4.validateMediaPairing(
                                        SIDECAR, Builder.newBuilder().build(), 1536));
        assertTrue(failure.getMessage().contains("no media projector"), failure.getMessage());
    }

    private static GGUF vision(int projectionWidth) {
        return Builder.newBuilder()
                .putString("clip.vision.projector_type", "gemma4v")
                .putInteger("clip.vision.projection_dim", projectionWidth)
                .build();
    }
}
