package com.qxotic.jinfer.testkit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.opentest4j.TestAbortedException;

class TestModelsTest {

    private static final String REF = "hf.co/ggml-org/stories15M_MOE:Q8_0";

    @TempDir Path root;

    @AfterEach
    void restoreCacheRoot() {
        System.clearProperty("jinfer.models");
    }

    @Test
    void aCachedModelResolves() throws IOException {
        System.setProperty("jinfer.models", root.toString());
        Path file = root.resolve("hf.co/ggml-org/stories15M_MOE/stories15M_MOE-Q8_0.gguf");
        Files.createDirectories(file.getParent());
        Files.writeString(file, "not really a model");
        assertEquals(file, TestModels.require(REF));
    }

    @Test
    void aMissingModelAbortsTheTestWithTheFix() {
        System.setProperty("jinfer.models", root.toString());
        var failure = assertThrows(TestAbortedException.class, () -> TestModels.require(REF));
        assertTrue(failure.getMessage().contains(REF), failure.getMessage());
        assertTrue(
                failure.getMessage().contains("scripts/download-models.sh"), failure.getMessage());
    }

    @Test
    void aBareRepoIsRefused() {
        var failure =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> TestModels.require("hf.co/ggml-org/stories15M_MOE"));
        assertTrue(failure.getMessage().contains(":Q8_0"), failure.getMessage());
    }

    @Test
    void aCompanionNamesItsExactFile() {
        // refused only if the pinning rule rejects it; the miss itself is a normal abort
        var failure =
                assertThrows(
                        TestAbortedException.class,
                        () -> {
                            System.setProperty("jinfer.models", root.toString());
                            TestModels.require("hf.co/ggml-org/stories15M_MOE/mmproj-F32.gguf");
                        });
        assertTrue(failure.getMessage().contains("mmproj-F32.gguf"), failure.getMessage());
    }
}
