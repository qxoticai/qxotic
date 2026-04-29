package com.qxotic.toknroll.gguf;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Regression tests for Tamil text tokenization.
 *
 * <p>These tests verify correct BPE merge behavior for byte-level encoded Tamil text. The Tamil
 * script uses multi-byte UTF-8 sequences that must be properly handled by the byte-level
 * encoder/decoder.
 *
 * <p><strong>llama.cpp discrepancy:</strong> llama.cpp's BPE tokenizer has a known bug where it
 * fails to apply certain valid merges for byte-level encoded non-ASCII text (e.g., Tamil,
 * Cyrillic). This causes llama.cpp to produce more tokens than expected for these scripts. Our
 * implementation correctly applies all valid merges from the GGUF file.
 *
 * <p>Related llama.cpp issues:
 *
 * <ul>
 *   <li><a href="https://github.com/ggml-org/llama.cpp/issues/21675">#21675</a> - Incorrect
 *       Cyrillic tokenization for MiniMax-M2 (same root cause: BPE merge algorithm mishandles
 *       non-ASCII byte-level encoded text)
 *   <li><a href="https://github.com/ggml-org/llama.cpp/issues/6809">#6809</a> - Multiple newlines
 *       don't merge into a single token (BPE merge bug)
 * </ul>
 *
 * <p>Example of the discrepancy for GPT-OSS 20B:
 *
 * <pre>
 *   Input:  "க்க" (Tamil "kka")
 *   Our impl:  [14719] (1 token - correctly merged)
 *   llama.cpp: [5465, 3647] (2 tokens - merge not applied)
 * </pre>
 */
@Tag("network")
@Tag("local-external")
class GGUFTokenizerTamilRegressionTest {

    @Test
    void gptOss20bTamilTextTokenization() {
        GGUFTokenizerLoader loader = GGUFTokenizerLoader.createBuilderWithBuiltins().build();
        Tokenizer tokenizer =
                loader.fromHuggingFace("unsloth", "gpt-oss-20b-GGUF", "gpt-oss-20b-Q8_0.gguf");

        // Single Tamil characters should tokenize to 1 token each
        assertTokenCount(tokenizer, "க", 1); // Tamil letter ka
        assertTokenCount(tokenizer, "க்", 1); // Tamil letter ka with pulli
        assertTokenCount(tokenizer, "ம்", 1); // Tamil letter ma with pulli

        // Combined forms that should merge
        assertTokenCount(tokenizer, "க்க", 1); // Should merge to single token

        // Full Tamil words
        assertTokenCount(tokenizer, "மதியிறுக்கம்", 6);
        assertTokenCount(tokenizer, "அரிஸ்டாட்டில்", 6);

        // Verify specific token IDs for "க்க"
        IntSequence tokens = tokenizer.encode("க்க");
        assertEquals(1, tokens.length(), "க்க should tokenize to 1 token");
        assertEquals(14719, tokens.intAt(0), "க்க should map to token 14719");

        // Verify decode round-trip
        String decoded = tokenizer.decode(tokens);
        assertEquals("க்க", decoded, "Decode should round-trip correctly");
    }

    @Test
    void gptOss20bTamilWordAutism() {
        GGUFTokenizerLoader loader = GGUFTokenizerLoader.createBuilderWithBuiltins().build();
        Tokenizer tokenizer =
                loader.fromHuggingFace("unsloth", "gpt-oss-20b-GGUF", "gpt-oss-20b-Q8_0.gguf");

        // "மதியிறுக்கம்" = Autism in Tamil
        // Should tokenize to: [ம, தி, யி, று, க்க, ம்]
        IntSequence tokens = tokenizer.encode("மதியிறுக்கம்");
        assertEquals(6, tokens.length(), "மதியிறுக்கம் should have 6 tokens");

        // Verify the 5th token is the merged form "க்க" (token 14719)
        assertEquals(14528, tokens.intAt(4), "5th token should be the merged க்க form");

        // Verify decode round-trip
        String decoded = tokenizer.decode(tokens);
        assertEquals("மதியிறுக்கம்", decoded, "Decode should round-trip correctly");
    }

    @Test
    void gptOss20bTamilWordAristotle() {
        GGUFTokenizerLoader loader = GGUFTokenizerLoader.createBuilderWithBuiltins().build();
        Tokenizer tokenizer =
                loader.fromHuggingFace("unsloth", "gpt-oss-20b-GGUF", "gpt-oss-20b-Q8_0.gguf");

        // "அரிஸ்டாட்டில்" = Aristotle in Tamil
        IntSequence tokens = tokenizer.encode("அரிஸ்டாட்டில்");
        assertEquals(6, tokens.length(), "அரிஸ்டாட்டில் should have 6 tokens");

        // Verify decode round-trip
        String decoded = tokenizer.decode(tokens);
        assertEquals("அரிஸ்டாட்டில்", decoded, "Decode should round-trip correctly");
    }

    private static void assertTokenCount(Tokenizer tokenizer, String text, int expectedCount) {
        IntSequence tokens = tokenizer.encode(text);
        assertEquals(
                expectedCount,
                tokens.length(),
                "\"" + text + "\" should tokenize to " + expectedCount + " token(s)");
    }
}
