package com.qxotic.jinfer.langchain4j;

import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * The {@code .model(String)} front door: one string that is a local path, a hub ref, or a URL. The
 * resolver itself is jinfer-hub's and tested there; what THIS module owes is that the builders
 * actually route through it - a local path string must load without touching the network, and a
 * string that names nothing must fail with the hub's teaching message, not an NPE at build().
 */
class ModelStringTest {

    @Test
    void aLocalPathStringResolvesWithoutTheNetwork(@TempDir Path dir) throws IOException {
        Path gguf = dir.resolve("tiny.gguf");
        Files.write(gguf, new byte[] {'G', 'G', 'U', 'F'});
        // resolution succeeds offline; the (invalid) GGUF then fails at load, which proves the
        // string reached the load path as a file
        assertThatThrownBy(() -> JinferChatModel.builder().model(gguf.toString()).build().close())
                .isInstanceOf(RuntimeException.class)
                .hasMessageNotContaining("no such model file");
    }

    @Test
    void resolutionHappensAtBuildNotInTheSetter() {
        // the setter RECORDS - the chain never blocks or throws; build() is where acquisition
        // lives, alongside the load, so there is one failure point and no download mid-chain
        JinferChatModel.Builder builder = JinferChatModel.builder().model("no/such/file.gguf");
        assertThatThrownBy(builder::build)
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("no such model file")
                .hasMessageContaining("hf.co/"); // the message teaches the ref grammar
    }

    @Test
    void theLastModelSetterWins(@TempDir Path dir) throws IOException {
        Path gguf = dir.resolve("tiny.gguf");
        Files.write(gguf, new byte[] {'G', 'G', 'U', 'F'});
        // the earlier ref is CLEARED, never fetched: build fails on the (invalid) local file,
        // which proves the path won - a hub message here would mean the ref was resolved
        assertThatThrownBy(
                        () ->
                                JinferChatModel.builder()
                                        .model("hf.co/nobody/nothing-GGUF")
                                        .modelPath(gguf)
                                        .build()
                                        .close())
                .isInstanceOf(RuntimeException.class)
                .hasMessageNotContaining("hf.co/nobody");

        // and the other way around: the string set last is the one that resolves
        assertThatThrownBy(
                        () ->
                                JinferChatModel.builder()
                                        .modelPath(gguf)
                                        .model("no/such/file.gguf")
                                        .build())
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("no such model file");
    }
}
