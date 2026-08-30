package com.qxotic.jinfer.models.qwen3;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.format.gguf.Builder;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** A retrieval-only Qwen3 checkpoint must fail a generative load with actionable advice. */
class Qwen3ProviderTest {

    @Test
    void claimsOnlyTheRetrievalArchitecture() {
        Qwen3Provider provider = new Qwen3Provider();
        assertTrue(provider.supports("qwen3"));
        assertFalse(provider.supports("qwen35"));
        assertFalse(provider.supports("qwen3moe"));
    }

    @Test
    void generativeLoadPointsToBothRetrievalEntryPoints(@TempDir Path directory)
            throws IOException {
        Path file = Files.createFile(directory.resolve("qwen3.gguf"));
        try (FileChannel channel = FileChannel.open(file)) {
            UnsupportedOperationException failure =
                    assertThrows(
                            UnsupportedOperationException.class,
                            () ->
                                    new Qwen3Provider()
                                            .load(
                                                    channel,
                                                    Builder.newBuilder().build(),
                                                    Arena.ofAuto(),
                                                    Map.of(),
                                                    null));
            assertTrue(failure.getMessage().contains("loadEmbedder"), failure.getMessage());
            assertTrue(failure.getMessage().contains("loadReranker"), failure.getMessage());
        }
    }
}
