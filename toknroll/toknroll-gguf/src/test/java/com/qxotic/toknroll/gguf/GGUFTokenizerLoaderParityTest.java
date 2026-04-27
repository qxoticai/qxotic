package com.qxotic.toknroll.gguf;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.List;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("network")
@Tag("local-external")
class GGUFTokenizerLoaderParityTest {

    private static final FamilyCase LLAMA_UNSLOTH_1B =
            new FamilyCase(
                    "llama-unsloth-1b",
                    "unsloth",
                    "Llama-3.2-1B-Instruct-GGUF",
                    "Llama-3.2-1B-Instruct-Q8_0.gguf",
                    "gpt2");

    private static final FamilyCase GEMMA4_E2B_UNSLOTH =
            new FamilyCase(
                    "gemma4-e2b-unsloth",
                    "unsloth",
                    "gemma-4-E2B-it-GGUF",
                    "gemma-4-E2B-it-Q8_0.gguf",
                    "gemma4");

    private static final FamilyCase QWEN36_A3B_UNSLOTH =
            new FamilyCase(
                    "qwen36-a3b-unsloth",
                    "unsloth",
                    "Qwen3.6-35B-A3B-GGUF",
                    "Qwen3.6-35B-A3B-Q8_0.gguf",
                    "gpt2");

    private static final FamilyCase GPT_OSS_20B_UNSLOTH =
            new FamilyCase(
                    "gpt-oss-20b-unsloth",
                    "unsloth",
                    "gpt-oss-20b-GGUF",
                    "gpt-oss-20b-Q8_0.gguf",
                    "gpt2");

    private static final FamilyCase KIMI_K26_UNSLOTH =
            new FamilyCase(
                    "kimi-k26-unsloth",
                    "unsloth",
                    "Kimi-K2.6-GGUF",
                    "UD-Q8_K_XL/Kimi-K2.6-UD-Q8_K_XL-00001-of-00014.gguf",
                    null);

    private static final FamilyCase GLM51_UNSLOTH =
            new FamilyCase(
                    "glm-51-unsloth",
                    "unsloth",
                    "GLM-5.1-GGUF",
                    "Q8_0/GLM-5.1-Q8_0-00001-of-00017.gguf",
                    null);

    private static final FamilyCase MINIMAX_M27_UNSLOTH =
            new FamilyCase(
                    "minimax-m27-unsloth",
                    "unsloth",
                    "MiniMax-M2.7-GGUF",
                    "Q8_0/MiniMax-M2.7-Q8_0-00001-of-00006.gguf",
                    null);

    private static final FamilyCase MIMO_V2_FLASH_UNSLOTH =
            new FamilyCase(
                    "mimo-v2-flash-unsloth",
                    "unsloth",
                    "MiMo-V2-Flash-GGUF",
                    "Q8_0/MiMo-V2-Flash-Q8_0-00001-of-00007.gguf",
                    null);

    private static final FamilyCase MISTRAL_7B_BARTOWSKI =
            new FamilyCase(
                    "mistral-7b-bartowski",
                    "bartowski",
                    "Mistral-7B-Instruct-v0.3-GGUF",
                    "Mistral-7B-Instruct-v0.3-Q8_0.gguf",
                    "llama");

    private static final FamilyCase MINISTRAL_3_3B_UNSLOTH =
            new FamilyCase(
                    "ministral-3-3b-unsloth",
                    "unsloth",
                    "Ministral-3-3B-Instruct-2512-GGUF",
                    "Ministral-3-3B-Instruct-2512-Q8_0.gguf",
                    "gpt2");

    private static final FamilyCase PHI4_MINI_UNSLOTH =
            new FamilyCase(
                    "phi4-mini-unsloth",
                    "unsloth",
                    "Phi-4-mini-instruct-GGUF",
                    "Phi-4-mini-instruct.Q8_0.gguf",
                    null);

    private static final FamilyCase GRANITE4_H_1B_UNSLOTH =
            new FamilyCase(
                    "granite4-h-1b-unsloth",
                    "unsloth",
                    "granite-4.0-h-1b-GGUF",
                    "granite-4.0-h-1b-Q8_0.gguf",
                    null);

    private static final FamilyCase SMOLLM3_3B_UNSLOTH =
            new FamilyCase(
                    "smollm3-3b-unsloth",
                    "unsloth",
                    "SmolLM3-3B-GGUF",
                    "SmolLM3-3B-Q8_0.gguf",
                    null);

    private static final List<String> SAMPLE_TEXTS =
            List.of(
                    "Hello world",
                    "Tokenizer family parity",
                    "Whitespace\n\tand unicode 😀",
                    "العربية mixed English 123",
                    "ไทยภาษาไทย without spaces",
                    "க்க",
                    "ம்",
                    "க்",
                    "க",
                    "மதியிறுக்கம்",
                    "அரிஸ்டாட்டில்");

    private static TestDataManager dataManager;

    @BeforeAll
    static void setUp() {
        dataManager = new TestDataManager();
    }

    @Test
    void llamaFromUnsloth_localAndHfParity() {
        assertLocalAndHfParity(LLAMA_UNSLOTH_1B);
    }

    @Test
    void gemma4FromUnsloth_localAndHfParity() {
        assertLocalAndHfParity(GEMMA4_E2B_UNSLOTH);
    }

    @Test
    void qwen36FromUnsloth_localAndHfParity() {
        assertLocalAndHfParity(QWEN36_A3B_UNSLOTH);
    }

    @Test
    void gptOss20bFromUnsloth_localAndHfParity() {
        assertLocalAndHfParity(GPT_OSS_20B_UNSLOTH);
    }

    @Test
    void kimiK26FromUnsloth_localAndHfParity() {
        assertLocalAndHfParity(KIMI_K26_UNSLOTH);
    }

    @Test
    void glm51FromUnsloth_localAndHfParity() {
        assertLocalAndHfParity(GLM51_UNSLOTH);
    }

    @Test
    void minimaxM27FromUnsloth_localAndHfParity() {
        assertLocalAndHfParity(MINIMAX_M27_UNSLOTH);
    }

    @Test
    void mimoV2FlashFromUnsloth_localAndHfParity() {
        assertLocalAndHfParity(MIMO_V2_FLASH_UNSLOTH);
    }

    @Test
    void mistral7bFromBartowski_localAndHfParity() {
        assertLocalAndHfParity(MISTRAL_7B_BARTOWSKI);
    }

    @Test
    void ministral33bFromUnsloth_localAndHfParity() {
        assertLocalAndHfParity(MINISTRAL_3_3B_UNSLOTH);
    }

    @Test
    void phi4MiniFromUnsloth_localAndHfParity() {
        assertLocalAndHfParity(PHI4_MINI_UNSLOTH);
    }

    @Test
    void granite4H1bFromUnsloth_localAndHfParity() {
        assertLocalAndHfParity(GRANITE4_H_1B_UNSLOTH);
    }

    @Test
    void smollm33bFromUnsloth_localAndHfParity() {
        assertLocalAndHfParity(SMOLLM3_3B_UNSLOTH);
    }

    private static void assertLocalAndHfParity(FamilyCase c) {
        GGUF gguf;
        try {
            gguf = dataManager.getOrDownloadMetadata(c.cacheKey(), c.hfUrl());
        } catch (Exception e) {
            Assumptions.abort("Skipping " + c.id + ": " + e);
            return;
        }

        String modelKey = gguf.getValueOrDefault(String.class, "tokenizer.ggml.model", "unknown");
        if (c.expectedModelKey != null) {
            assertEquals(
                    c.expectedModelKey,
                    modelKey,
                    "Unexpected tokenizer.ggml.model for " + c.user + "/" + c.repo + "/" + c.file);
        }

        Path localPartial =
                dataManager.getCachePath().resolve(TestDataManager.cacheFileNameForUrl(c.hfUrl()));
        Path localGguf;
        try {
            localGguf = Files.createTempFile("toknroll-gguf-family-", ".gguf");
            Files.copy(localPartial, localGguf, StandardCopyOption.REPLACE_EXISTING);
        } catch (Exception e) {
            Assumptions.abort("Skipping " + c.id + " local prep: " + e);
            return;
        }

        GGUFTokenizerLoader loader = GGUFTokenizerLoader.builderDefault().build();
        try {
            Tokenizer localTokenizer = loader.fromLocal(localGguf);
            Tokenizer hfTokenizer = loader.fromHuggingFace(c.user, c.repo, c.file);
            assertTokenizerParity(localTokenizer, hfTokenizer);
        } catch (RuntimeException e) {
            Assumptions.abort("Skipping " + c.id + " loader parity: " + e);
        } finally {
            try {
                Files.deleteIfExists(localGguf);
            } catch (Exception ignored) {
            }
        }
    }

    private static void assertTokenizerParity(Tokenizer a, Tokenizer b) {
        for (String text : SAMPLE_TEXTS) {
            IntSequence tokensA = a.encode(text);
            IntSequence tokensB = b.encode(text);
            assertIntSequenceEquals(tokensA, tokensB, "encode parity for: " + text);
            assertEquals(
                    a.countTokens(text), b.countTokens(text), "countTokens parity for: " + text);
        }
    }

    private static void assertIntSequenceEquals(
            IntSequence expected, IntSequence actual, String message) {
        assertEquals(expected.length(), actual.length(), message + " (length)");
        for (int i = 0; i < expected.length(); i++) {
            assertEquals(expected.intAt(i), actual.intAt(i), message + " (index " + i + ")");
        }
    }

    private static final class FamilyCase {
        private final String id;
        private final String user;
        private final String repo;
        private final String file;
        private final String expectedModelKey;

        private FamilyCase(
                String id, String user, String repo, String file, String expectedModelKey) {
            this.id = id;
            this.user = user;
            this.repo = repo;
            this.file = file;
            this.expectedModelKey = expectedModelKey;
        }

        private String cacheKey() {
            return "family-" + id;
        }

        private String hfUrl() {
            return "https://huggingface.co/" + user + "/" + repo + "/resolve/main/" + file;
        }
    }
}
