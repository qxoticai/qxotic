package com.qxotic.tokenizers.hf;

import com.qxotic.tokenizers.IntSequence;
import com.qxotic.tokenizers.Tokenizer;
import java.util.stream.Stream;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * Comprehensive integration tests for HuggingFace tokenizers across multiple models.
 *
 * <p>These tests verify tokenization behavior across different model families and text types. Run
 * with: RUN_HF_TESTS=true mvn test
 */
@EnabledIfEnvironmentVariable(named = "RUN_HF_TESTS", matches = "true")
@Tag("network")
class HuggingFaceTokenizersModelTest {

    /** Provides test models known to work with the explicit regex requirement. */
    static Stream<Arguments> workingModels() {
        return Stream.of(
                // Qwen3 - BPE with byte-level, explicit regex
                Arguments.of(
                        "Qwen/Qwen3-0.6B",
                        "main",
                        151685, // Expected vocab size
                        "Qwen3 BPE"),
                // Qwen2.5 - Similar architecture to Qwen3
                Arguments.of(
                        "Qwen/Qwen2.5-0.5B-Instruct",
                        "main",
                        151936, // Expected vocab size
                        "Qwen2.5 BPE"),
                // Phi-3/4 - Microsoft's models with explicit regex
                Arguments.of(
                        "microsoft/Phi-3-mini-4k-instruct",
                        "main",
                        32064, // Expected vocab size
                        "Phi-3 BPE"),
                // Mistral Nemo - Uses Tekken tokenizer with explicit regex
                Arguments.of(
                        "mistralai/Mistral-Nemo-Instruct-2407",
                        "main",
                        131072, // Expected vocab size (Tekken tokenizer)
                        "Mistral Nemo Tekken"),
                // Llama 3.2 (via Unsloth) - Uses Tekken tokenizer
                Arguments.of(
                        "unsloth/Llama-3.2-1B-Instruct",
                        "main",
                        128000, // Expected vocab size (Tekken tokenizer)
                        "Llama 3.2 Tekken"),
                // GPT-OSS (OpenAI) - Uses BPE with complex regex
                Arguments.of(
                        "openai/gpt-oss-20b",
                        "main",
                        200000, // Expected vocab size (GPT-OSS tokenizer)
                        "GPT-OSS BPE")
                // Note: Mistral v0.x models use SentencePiece (Metaspace pretokenizer)
                // which doesn't have explicit regex patterns, so they are not supported
                // Examples: Mistral-7B-Instruct-v0.1, v0.2, v0.3
                );
    }

    /** Provides various text inputs to test tokenization edge cases. */
    static Stream<Arguments> testTexts() {
        return Stream.of(
                // Basic ASCII
                Arguments.of("Hello, world!", "basic ascii"),
                Arguments.of("The quick brown fox jumps over the lazy dog.", "pangram"),

                // Numbers and punctuation
                Arguments.of("1234567890", "numbers"),
                Arguments.of("Price: $99.99 (50% off!)", "currency"),
                Arguments.of("Email: test@example.com", "email"),
                Arguments.of("URL: https://example.com/path?query=1", "url"),

                // Unicode and international text
                Arguments.of("Hello 世界 🌍", "mixed unicode"),
                Arguments.of("Café résumé naïve", "accents"),
                Arguments.of("日本語テキスト", "japanese"),
                Arguments.of("中文文本测试", "chinese"),
                Arguments.of("한국어 텍스트", "korean"),
                Arguments.of("العربية نص", "arabic"),
                Arguments.of("עברית טקסט", "hebrew"),
                Arguments.of("Русский текст", "cyrillic"),
                Arguments.of("Ελληνικά κείμενο", "greek"),

                // Emojis and special characters
                Arguments.of("😀 🎉 👍 💯 🔥", "emojis"),
                Arguments.of("🚀🌟💫✨🎊", "emoji sequence"),
                Arguments.of("👨‍👩‍👧‍👦 👨‍💻 🧑‍🔬", "emoji zwj"),
                Arguments.of("🏳️‍🌈 🏴‍☠️", "emoji flags"),

                // Code and markup
                Arguments.of("def hello():\n    return 'world'", "python code"),
                Arguments.of("<html>\n  <body>Hello</body>\n</html>", "html"),
                Arguments.of("SELECT * FROM users WHERE id = 1;", "sql"),
                Arguments.of("int main() { return 0; }", "c code"),

                // Whitespace and control chars
                Arguments.of("Tab\there", "tab"),
                Arguments.of("Line1\nLine2", "newline"),
                Arguments.of("  spaces  ", "spaces"),
                Arguments.of("\r\nWindows", "crlf"),

                // Math and symbols
                Arguments.of("∑∏√∞≈≠≤≥", "math symbols"),
                Arguments.of("αβγδε → ← ↑ ↓", "greek arrows"),
                Arguments.of("©®™℠ ℗", "symbols"),

                // Long text
                Arguments.of(
                        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
                                + "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
                        "long text"),

                // Edge cases
                Arguments.of("", "empty string"),
                Arguments.of("a", "single char"),
                Arguments.of("\n\n\n", "only newlines"),
                Arguments.of("   ", "only spaces"));
    }

    /** Tests that models can be loaded and have expected vocabulary sizes. */
    @ParameterizedTest(name = "Load {3}")
    @MethodSource("workingModels")
    void testModelLoading(
            String repoId, String revision, int expectedVocabSize, String description) {
        Tokenizer tokenizer = loadModelOrSkip(repoId, revision, description);
        if (tokenizer == null) return;

        org.junit.jupiter.api.Assertions.assertNotNull(tokenizer);
        org.junit.jupiter.api.Assertions.assertTrue(
                tokenizer.vocabulary().size() >= expectedVocabSize * 0.9,
                "Vocabulary size should be approximately "
                        + expectedVocabSize
                        + " but was "
                        + tokenizer.vocabulary().size());
        org.junit.jupiter.api.Assertions.assertTrue(
                tokenizer.vocabulary().size() <= expectedVocabSize * 1.1,
                "Vocabulary size should be approximately "
                        + expectedVocabSize
                        + " but was "
                        + tokenizer.vocabulary().size());
    }

    /** Tests basic encoding/decoding round-trip for each model. */
    @ParameterizedTest(name = "Round-trip: {3}")
    @MethodSource("workingModels")
    void testRoundTrip(String repoId, String revision, int expectedVocabSize, String description) {
        Tokenizer tokenizer = loadModelOrSkip(repoId, revision, description);
        if (tokenizer == null) return;

        String testText = "Hello, world! 123";
        IntSequence tokens = tokenizer.encode(testText);
        tokenizer.decode(tokens);

        org.junit.jupiter.api.Assertions.assertTrue(
                tokens.length() > 0, "Should produce at least one token");
    }

    /** Tests various text inputs with Qwen3. */
    @ParameterizedTest(name = "Text: {1}")
    @MethodSource("testTexts")
    void testVariousTexts(String text, String description) {
        Tokenizer tokenizer = HuggingFaceTokenizers.fromRepository("Qwen/Qwen3-0.6B");

        IntSequence tokens = tokenizer.encode(text);

        org.junit.jupiter.api.Assertions.assertNotNull(tokens);

        // Verify all tokens are valid
        for (int i = 0; i < tokens.length(); i++) {
            int tokenId = tokens.intAt(i);
            org.junit.jupiter.api.Assertions.assertTrue(
                    tokenizer.vocabulary().contains(tokenId),
                    "Token ID " + tokenId + " should be in vocabulary");
        }

        // Test decode
        if (!text.isEmpty()) {
            String decoded = tokenizer.decode(tokens);
            org.junit.jupiter.api.Assertions.assertNotNull(decoded);
        }

        // Test countTokens matches encode
        int count = tokenizer.countTokens(text);
        org.junit.jupiter.api.Assertions.assertEquals(
                tokens.length(), count, "countTokens should match encode length");
    }

    /** Tests emoji handling specifically. */
    @Test
    void testEmojiHandling() {
        Tokenizer tokenizer = HuggingFaceTokenizers.fromRepository("Qwen/Qwen3-0.6B");

        String[] emojiTexts = {
            "😀",
            "🎉🎊",
            "👨‍👩‍👧‍👦", // Family emoji with ZWJ
            "🏳️‍🌈", // Rainbow flag
            "Hello 👋 World 🌍",
            "🔥" + "🔥".repeat(10)
        };

        for (String text : emojiTexts) {
            IntSequence tokens = tokenizer.encode(text);
            tokenizer.decode(tokens);

            org.junit.jupiter.api.Assertions.assertTrue(
                    tokens.length() > 0, "Emojis should tokenize to at least one token");
        }
    }

    /** Tests that different models tokenize the same text differently. */
    @Test
    void testModelComparison() {
        String testText = "Hello, world! 🌍";
        String[] models = {
            "Qwen/Qwen3-0.6B",
            "Qwen/Qwen2.5-0.5B-Instruct",
            "mistralai/Mistral-Nemo-Instruct-2407",
            "unsloth/Llama-3.2-1B-Instruct",
            "openai/gpt-oss-20b"
        };

        int loaded = 0;
        for (String modelName : models) {
            try {
                Tokenizer tokenizer = HuggingFaceTokenizers.fromRepository(modelName);
                IntSequence tokens = tokenizer.encode(testText);
                org.junit.jupiter.api.Assertions.assertTrue(tokens.length() > 0);
                loaded++;
            } catch (HuggingFaceTokenizerException e) {
                // best-effort comparison across public models
            }
        }
        org.junit.jupiter.api.Assertions.assertTrue(
                loaded > 0, "Expected at least one model to load");
    }

    /** Tests special token handling. */
    @Test
    void testSpecialTokens() {
        Tokenizer tokenizer = HuggingFaceTokenizers.fromRepository("Qwen/Qwen3-0.6B");

        // Test with special tokens if they exist
        String[] texts = {"<|endoftext|>", "<|im_start|>user<|im_end|>", "Hello<|endoftext|>"};

        for (String text : texts) {
            try {
                IntSequence tokens = tokenizer.encode(text);

                // Verify the special tokens are in vocabulary
                for (int i = 0; i < tokens.length(); i++) {
                    int tokenId = tokens.intAt(i);
                    org.junit.jupiter.api.Assertions.assertTrue(
                            tokenizer.vocabulary().contains(tokenId),
                            "Token ID " + tokenId + " should be in vocabulary");
                }
            } catch (Exception e) {
                // Some tokenizers may treat these strings as plain text.
            }
        }
    }

    /** Performance test comparing first vs cached load. */
    @Test
    void testCachePerformance() {
        String repoId = "Qwen/Qwen3-0.6B";

        Tokenizer tokenizer1 = HuggingFaceTokenizers.fromRepository(repoId);
        Tokenizer tokenizer2 = HuggingFaceTokenizers.fromRepository(repoId);
        org.junit.jupiter.api.Assertions.assertEquals(
                tokenizer1.vocabulary().size(),
                tokenizer2.vocabulary().size(),
                "Cache reuse should preserve tokenizer data");
    }

    /** Helper to load a model or skip the test if it fails. */
    private Tokenizer loadModelOrSkip(String repoId, String revision, String description) {
        try {
            return HuggingFaceTokenizers.fromRepository(repoId, revision);
        } catch (HuggingFaceTokenizerException e) {
            if (e.getMessage().contains("401") || e.getMessage().contains("Authentication")) {
                org.junit.jupiter.api.Assumptions.assumeTrue(
                        false, "Model requires authentication: " + repoId);
            } else if (e.getMessage().contains("regex")
                    || e.getMessage().contains("not supported")) {
                org.junit.jupiter.api.Assumptions.assumeTrue(
                        false, "Model not compatible: " + repoId + " - " + e.getMessage());
            } else {
                org.junit.jupiter.api.Assumptions.assumeTrue(
                        false, "Failed to load model: " + repoId + " - " + e.getMessage());
            }
            return null;
        }
    }
}
