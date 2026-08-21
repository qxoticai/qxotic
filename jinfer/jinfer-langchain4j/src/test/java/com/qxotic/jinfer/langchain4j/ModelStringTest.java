package com.qxotic.jinfer.langchain4j;

import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * The builder's two model doors are explicit: {@code .model(String)} is a model ref, {@code
 * .modelPath(Path)} is local. This module only owes that the boundary is enforced and the remote
 * ref is resolved at build time, not in the setter.
 */
class ModelStringTest {

    @Test
    void aLocalPathStringIsRefusedAndPointsToModelPath(@TempDir Path dir) throws Exception {
        Path gguf = dir.resolve("tiny.gguf");
        Files.write(gguf, new byte[] {'G', 'G', 'U', 'F'});
        assertThatThrownBy(() -> JinferChatModel.builder().model(gguf.toString()))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("modelPath");
    }

    @Test
    void aUrlIsRefusedAndPointsToModelPath() {
        assertThatThrownBy(() -> JinferChatModel.builder().model("https://example.org/model.gguf"))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("URL")
                .hasMessageContaining("modelPath");
    }

    @Test
    void aBareRepoIsToldItsMissingHost() {
        assertThatThrownBy(() -> JinferChatModel.builder().model("unsloth/Qwen3.5-4B-GGUF:Q4_K_M"))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("missing its host")
                .hasMessageContaining("hf.co/unsloth/Qwen3.5-4B-GGUF:Q4_K_M");
    }

    @Test
    void aLocalPathStillResolvesThroughModelPath(@TempDir Path dir) throws Exception {
        Path gguf = dir.resolve("tiny.gguf");
        Files.write(gguf, new byte[] {'G', 'G', 'U', 'F'});
        // the path reaches the load stage; the invalid GGUF then fails there, proving it was NOT
        // treated as a remote ref
        assertThatThrownBy(() -> JinferChatModel.builder().modelPath(gguf).build().close())
                .isInstanceOf(RuntimeException.class)
                .hasMessageNotContaining("no such model file");
    }

    @Test
    void aRemoteStringIsResolvedAtBuildNotInTheSetter() {
        JinferChatModel.Builder builder =
                JinferChatModel.builder().model("hf.co/nobody/nothing-GGUF:Q4_K_M");
        assertThatThrownBy(builder::build).isInstanceOf(RuntimeException.class);
    }
}
