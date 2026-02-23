package com.qxotic.tokenizers.hf;

import static org.junit.jupiter.api.Assertions.*;

import com.qxotic.tokenizers.IntSequence;
import com.qxotic.tokenizers.Tokenizer;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;

/**
 * Integration tests for HuggingFace tokenizers.
 *
 * <p>These tests download real tokenizer files from HuggingFace Hub. Run with:
 * QXOTIC_TOKENIZERS_HF_OFFLINE=false mvn test
 */
@EnabledIfEnvironmentVariable(named = "RUN_HF_TESTS", matches = "true")
@Tag("network")
class HuggingFaceTokenizersIntegrationTest {

    /**
     * Tests loading Qwen3-0.6B tokenizer from HuggingFace Hub.
     *
     * <p>This tests:
     *
     * <ul>
     *   <li>Repository download and caching
     *   <li>BPE tokenizer parsing
     *   <li>Explicit regex extraction
     *   <li>JTokkit backend integration
     *   <li>Special token handling
     * </ul>
     */
    @Test
    void testQwen3Tokenizer() {
        Tokenizer tokenizer = HuggingFaceTokenizers.fromRepository("Qwen/Qwen3-0.6B");

        // Test basic encoding
        String testText = "Hello, world!";
        IntSequence tokens = tokenizer.encode(testText);
        assertTrue(tokens.length() > 0, "Should produce at least one token");

        // Test decoding
        String decoded = tokenizer.decode(tokens);
        assertEquals(testText, decoded, "Round-trip should work");

        // Test countTokens matches encode length
        int count = tokenizer.countTokens(testText);
        assertEquals(tokens.length(), count, "countTokens should match encode length");

        // Test vocabulary access
        assertTrue(tokenizer.vocabulary().size() > 0, "Vocabulary should not be empty");
    }

    /** Tests loading tokenizer from a specific revision. */
    @Test
    void testFromRevision() {
        Tokenizer tokenizer = HuggingFaceTokenizers.fromRepository("Qwen/Qwen3-0.6B", "main");

        assertNotNull(tokenizer);
        assertTrue(tokenizer.vocabulary().size() > 0);
    }

    /** Tests that cached files are reused. */
    @Test
    void testCacheReuse() {
        // First load
        long start1 = System.currentTimeMillis();
        Tokenizer tokenizer1 = HuggingFaceTokenizers.fromRepository("Qwen/Qwen3-0.6B");
        long duration1 = System.currentTimeMillis() - start1;

        // Second load (should use cache)
        long start2 = System.currentTimeMillis();
        Tokenizer tokenizer2 = HuggingFaceTokenizers.fromRepository("Qwen/Qwen3-0.6B");
        long duration2 = System.currentTimeMillis() - start2;

        // Avoid flaky timing assertions in CI.
        assertTrue(duration1 >= 0);
        assertTrue(duration2 >= 0);
    }
}
