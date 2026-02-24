package com.qxotic.tokenizers.testkit;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.tokenizers.IntSequence;
import com.qxotic.tokenizers.Tokenizer;
import java.nio.charset.StandardCharsets;

public final class TokenizerAssertions {

    private TokenizerAssertions() {}

    public static void assertRoundTrip(Tokenizer tokenizer, String text, String context) {
        IntSequence tokens = tokenizer.encode(text);
        String decoded = tokenizer.decode(tokens);
        assertEquals(text, decoded, context + " round-trip");
    }

    public static void assertCountMatchesEncode(Tokenizer tokenizer, String text, String context) {
        IntSequence tokens = tokenizer.encode(text);
        assertEquals(
                tokens.length(), tokenizer.countTokens(text), context + " count matches encode");
    }

    public static void assertTokensInVocabulary(
            Tokenizer tokenizer, IntSequence tokens, String context) {
        for (int i = 0; i < tokens.length(); i++) {
            assertTrue(
                    tokenizer.vocabulary().contains(tokens.intAt(i)),
                    context + " token in vocab @" + i);
        }
    }

    public static void assertDecodeBytesUtf8(Tokenizer tokenizer, String text, String context) {
        IntSequence tokens = tokenizer.encode(text);
        byte[] decodedBytes = tokenizer.decodeBytes(tokens);
        assertArrayEquals(
                text.getBytes(StandardCharsets.UTF_8), decodedBytes, context + " decodeBytes");
    }

    public static void assertRoundTripSamples(
            Tokenizer tokenizer, String[] samples, String context) {
        for (String sample : samples) {
            IntSequence tokens = tokenizer.encode(sample);
            if (sample.isEmpty()) {
                assertEquals(0, tokens.length(), context + " empty text");
            } else {
                assertTrue(tokens.length() > 0, context + " tokens for " + sample);
            }
            assertEquals(sample, tokenizer.decode(tokens), context + " round-trip for " + sample);
        }
    }
}
