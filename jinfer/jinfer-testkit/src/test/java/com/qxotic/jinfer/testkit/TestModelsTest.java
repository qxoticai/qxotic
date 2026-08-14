package com.qxotic.jinfer.testkit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Optional;
import java.util.function.Function;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.opentest4j.TestAbortedException;

class TestModelsTest {

    private static final String REF = "hf.co/ggml-org/stories15M_MOE:Q8_0";
    private static final String OVERRIDE_KEY = "jinfer.testModel.stories15M_MOE:Q8_0";

    @TempDir Path root;

    /** A store with nothing cached. */
    private static Optional<Path> miss(String ref) {
        return Optional.empty();
    }

    /** A property source answering one key, null elsewhere - no process-global mutation. */
    private static Function<String, String> props(String key, String value) {
        return k -> k.equals(key) ? value : null;
    }

    @Test
    void anOverrideServesTheRef() throws IOException {
        Path file = Files.writeString(root.resolve("my-own-quant.gguf"), "not really a model");
        Function<String, String> props = props(OVERRIDE_KEY, file.toString());
        assertEquals(file, TestModels.resolve(REF, TestModelsTest::miss, props).orElseThrow());
    }

    @Test
    void aStaleOverrideFailsLoudly() {
        // an explicit pointer that resolves to nothing is a tester error, never a silent skip
        var failure =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                TestModels.resolve(
                                        REF,
                                        TestModelsTest::miss,
                                        props(OVERRIDE_KEY, "/no/such/file.gguf")));
        assertTrue(failure.getMessage().contains(OVERRIDE_KEY), failure.getMessage());
    }

    @Test
    void aFileFormRefKeysOnItsFileName() throws IOException {
        Path file = Files.writeString(root.resolve("my-mmproj.gguf"), "not really a model");
        assertEquals(
                Optional.of(file),
                TestModels.resolve(
                        "hf.co/ggml-org/stories15M_MOE/mmproj-F32.gguf",
                        TestModelsTest::miss,
                        props("jinfer.testModel.mmproj-F32.gguf", file.toString())));
    }

    @Test
    void aCachedModelResolves() throws IOException {
        Path file = Files.writeString(root.resolve("stories15M_MOE-Q8_0.gguf"), "not a model");
        assertEquals(file, TestModels.require(REF, ref -> Optional.of(file), key -> null));
    }

    @Test
    void aMissingModelAbortsTheTestWithTheFix() {
        var failure =
                assertThrows(
                        TestAbortedException.class,
                        () -> TestModels.require(REF, TestModelsTest::miss, key -> null));
        assertTrue(failure.getMessage().contains(REF), failure.getMessage());
        assertTrue(
                failure.getMessage().contains("scripts/download-models.sh"), failure.getMessage());
    }

    @Test
    void aBareRepoIsRefused() {
        var failure =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                TestModels.require(
                                        "hf.co/ggml-org/stories15M_MOE",
                                        TestModelsTest::miss,
                                        key -> null));
        assertTrue(failure.getMessage().contains(":Q8_0"), failure.getMessage());
    }

    @Test
    void aCompanionNamesItsExactFile() {
        // passes the pinning rule (names its exact file); the miss itself is a normal abort
        var failure =
                assertThrows(
                        TestAbortedException.class,
                        () ->
                                TestModels.require(
                                        "hf.co/ggml-org/stories15M_MOE/mmproj-F32.gguf",
                                        TestModelsTest::miss,
                                        key -> null));
        assertTrue(failure.getMessage().contains("mmproj-F32.gguf"), failure.getMessage());
    }
}
