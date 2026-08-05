package com.qxotic.toknroll.gguf;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class GGUFTokenizerLoaderApiTest {

    @Test
    void builderMethodsCreateLoaders() {
        assertNotNull(GGUFTokenizerLoader.createBuilderWithBuiltins().build());
        assertNotNull(GGUFTokenizerLoader.createEmptyBuilder().build());
    }

    /**
     * Pins builtin coverage of the common llama.cpp pre-tokenizer names (llama-vocab.cpp). Aliasing
     * TO a name only succeeds when that name is registered, so this walks the public API.
     */
    @Test
    void builtinsCoverCommonLlamaCppPreTokenizers() {
        String[] names = {
            "gpt-2",
            "gpt2",
            "granite-docling",
            "exaone4",
            "modern-bert",
            "llama3",
            "llama-v3",
            "llama-bpe",
            "pixtral",
            "smollm3",
            "llama4",
            "glm4",
            "dbrx",
            "smaug-bpe",
            "falcon3",
            "falcon-h1",
            "jina-v5-nano",
            "qwen2",
            "solar-open",
            "hunyuan",
            "grok-2",
            "deepseek-r1-qwen",
            "qwen35",
            "lfm2",
            "tekken",
            "gpt-4o",
            "kanana2",
            "minimax-m2",
            "kimi-k2",
            "gemma4",
            "granite-embed-multi-311m",
            "deepseek-v3",
            "smollm",
            "command-r",
            "exaone",
            "minicpm5",
            "default",
        };
        GGUFTokenizerLoader.Builder builder = GGUFTokenizerLoader.createBuilderWithBuiltins();
        for (String name : names) {
            builder.aliasPreTokenizer("probe-" + name, name);
        }
        assertNotNull(builder.build());
    }

    @Test
    void aliasPreTokenizerRejectsUnknownTarget() {
        GGUFTokenizerLoader.Builder builder = GGUFTokenizerLoader.createBuilderWithBuiltins();
        IllegalArgumentException e =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> builder.aliasPreTokenizer("yi", "no-such-scheme"));
        assertTrue(e.getMessage().contains("llama-bpe"));
    }

    @Test
    void fromLocalRejectsNonGgufFile(@TempDir Path tempDir) throws Exception {
        GGUFTokenizerLoader loader = GGUFTokenizerLoader.createBuilderWithBuiltins().build();
        Path file = tempDir.resolve("not-gguf.txt");
        Files.writeString(file, "content");
        assertThrows(IllegalArgumentException.class, () -> loader.fromLocal(file));
    }

    @Test
    void fromHuggingFaceRequiresExactGgufPath() {
        GGUFTokenizerLoader loader = GGUFTokenizerLoader.createBuilderWithBuiltins().build();
        assertThrows(
                RuntimeException.class,
                () -> loader.fromHuggingFace("unsloth", "nonexistent", "not-a-gguf.txt"));
    }
}
