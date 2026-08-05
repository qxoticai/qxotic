package com.qxotic.jinfer.models.gemma4;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

/**
 * The mmproj pairing contract against real sidecar headers: the right sidecar passes, a sidecar
 * from another Gemma 4 size fails at load naming both dims and the remedy - never at the first
 * media request with a shape error. Header-only reads; skipped when the local models are absent.
 */
class MmprojPairingTest {

    private static final Path E2B_MMPROJ =
            Path.of(
                    "/home/mukel/Desktop/playground/models/unsloth/gemma-4-E2B-it-GGUF/mmproj-F32.gguf");
    private static final Path B12_MMPROJ =
            Path.of(
                    "/home/mukel/Desktop/playground/models/unsloth/gemma-4-12b-it-GGUF/mmproj-F32.gguf");

    @Test
    void theRightSidecarPasses() {
        Assumptions.assumeTrue(Files.exists(E2B_MMPROJ));
        assertDoesNotThrow(() -> Gemma4.validateMmproj(E2B_MMPROJ, 1536));
    }

    @Test
    void aSidecarFromAnotherSizeFailsWithTheRemedy() {
        Assumptions.assumeTrue(Files.exists(E2B_MMPROJ));
        // the E2B sidecar (1536) offered to a 12b-sized model (3840)
        IllegalArgumentException e =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> Gemma4.validateMmproj(E2B_MMPROJ, 3840));
        assertTrue(e.getMessage().contains("1536"), e.getMessage());
        assertTrue(e.getMessage().contains("3840"), e.getMessage());
        assertTrue(e.getMessage().contains("different Gemma 4 size"), e.getMessage());
    }

    @Test
    void theTwelveBSidecarRejectsAnE2bModel() {
        Assumptions.assumeTrue(Files.exists(B12_MMPROJ));
        IllegalArgumentException e =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> Gemma4.validateMmproj(B12_MMPROJ, 1536));
        assertTrue(e.getMessage().contains("different Gemma 4 size"), e.getMessage());
    }
}
