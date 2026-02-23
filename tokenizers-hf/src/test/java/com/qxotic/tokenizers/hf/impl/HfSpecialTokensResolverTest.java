package com.qxotic.tokenizers.hf.impl;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.tokenizers.hf.HuggingFaceTokenizerException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class HfSpecialTokensResolverTest {

    @TempDir Path tempDir;

    @Test
    void resolveMergesTokenizerAndConfigTokens() throws IOException {
        Path tokenizerJson = tempDir.resolve("tokenizer.json");
        Path tokenizerConfig = tempDir.resolve("tokenizer_config.json");

        Files.writeString(tokenizerJson, "{}");
        Files.writeString(
                tokenizerConfig,
                """
                {
                  "added_tokens_decoder": {
                    "101": {"content": "<|eot|>"},
                    "102": {"content": "<|eom|>"}
                  }
                }
                """);

        HfSpecialTokensResolver resolver =
                new HfSpecialTokensResolver(tokenizerJson, tokenizerConfig);
        Map<String, Integer> resolved = resolver.resolve(Map.of("<|bos|>", 100));

        assertEquals(3, resolved.size());
        assertEquals(100, resolved.get("<|bos|>"));
        assertEquals(101, resolved.get("<|eot|>"));
        assertEquals(102, resolved.get("<|eom|>"));
    }

    @Test
    void resolveFailsOnTokenConflict() throws IOException {
        Path tokenizerJson = tempDir.resolve("tokenizer.json");
        Path tokenizerConfig = tempDir.resolve("tokenizer_config.json");

        Files.writeString(tokenizerJson, "{}");
        Files.writeString(
                tokenizerConfig,
                """
                {
                  "added_tokens_decoder": {
                    "200": {"content": "<|eot|>"}
                  }
                }
                """);

        HfSpecialTokensResolver resolver =
                new HfSpecialTokensResolver(tokenizerJson, tokenizerConfig);
        HuggingFaceTokenizerException ex =
                assertThrows(
                        HuggingFaceTokenizerException.class,
                        () -> resolver.resolve(Map.of("<|eot|>", 100)));

        assertTrue(ex.getMessage().contains("Conflicting token-to-ID mapping"));
    }

    @Test
    void resolveFailsOnDuplicateIdConflict() throws IOException {
        Path tokenizerJson = tempDir.resolve("tokenizer.json");
        Path tokenizerConfig = tempDir.resolve("tokenizer_config.json");

        Files.writeString(tokenizerJson, "{}");
        Files.writeString(
                tokenizerConfig,
                """
                {
                  "added_tokens_decoder": {
                    "100": {"content": "<|eot|>"}
                  }
                }
                """);

        HfSpecialTokensResolver resolver =
                new HfSpecialTokensResolver(tokenizerJson, tokenizerConfig);
        HuggingFaceTokenizerException ex =
                assertThrows(
                        HuggingFaceTokenizerException.class,
                        () -> resolver.resolve(Map.of("<|bos|>", 100)));

        assertTrue(ex.getMessage().contains("Duplicate token ID"));
    }
}
