package com.qxotic.jinfer.hub;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Optional;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * {@link ModelStore#find} against a planted flat cache. Untagged on purpose: find never touches the
 * network, so these gates run in the default build - a miss that tried to list a repository would
 * throw here, not skip.
 */
class ModelStoreFindTest {

    private static final String REF = "hf.co/ggml-org/stories15M_MOE:Q8_0";

    @TempDir Path root;

    @AfterEach
    void restoreCacheRoot() {
        System.clearProperty("jinfer.models");
    }

    private Path plant(String repoRelative) throws IOException {
        System.setProperty("jinfer.models", root.toString());
        Path file = root.resolve("hf.co/ggml-org/stories15M_MOE").resolve(repoRelative);
        Files.createDirectories(file.getParent());
        return Files.writeString(file, "not really a model");
    }

    @Test
    void aQuantRefFindsTheMatchingFile() throws IOException {
        Path file = plant("stories15M_MOE-Q8_0.gguf");
        assertEquals(Optional.of(file), ModelStore.find(REF));
    }

    @Test
    void anExplicitFileRefFindsThatFile() throws IOException {
        Path file = plant("mmproj-F32.gguf");
        assertEquals(
                Optional.of(file),
                ModelStore.find("hf.co/ggml-org/stories15M_MOE/mmproj-F32.gguf"));
    }

    @Test
    void aMissIsEmptyWithoutAnyListing() {
        System.setProperty("jinfer.models", root.toString());
        assertEquals(Optional.empty(), ModelStore.find(REF));
    }

    @Test
    void aDifferentQuantIsAMiss() throws IOException {
        plant("stories15M_MOE-F16.gguf");
        assertEquals(Optional.empty(), ModelStore.find(REF));
    }

    @Test
    void anAmbiguousCacheIsAMissRatherThanAGuess() throws IOException {
        plant("stories15M_MOE-a-Q8_0.gguf");
        plant("stories15M_MOE-b-Q8_0.gguf");
        assertEquals(Optional.empty(), ModelStore.find(REF));
    }

    @Test
    void aLocalPathPassesThrough() throws IOException {
        Path local = Files.writeString(root.resolve("local-model.gguf"), "not really a model");
        assertEquals(Optional.of(local), ModelStore.find(local.toString()));
        assertEquals(Optional.empty(), ModelStore.find(root.resolve("absent.gguf").toString()));
    }

    @Test
    void aPlainUrlIsRefused() {
        var failure =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> ModelStore.find("https://example.org/models/x.gguf"));
        assertTrue(failure.getMessage().contains("checksum"), failure.getMessage());
    }
}
