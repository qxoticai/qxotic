package com.qxotic.jinfer.llm;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.format.gguf.Builder;
import com.qxotic.format.gguf.GGUF;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Toknroll;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

/**
 * The facade's jinfer-owned slice: the id-space guard, the override-detection fact, and the loud
 * refusal of the moved {@code jinfer.preTokenizer.*} namespace. Scheme resolution itself is
 * toknroll's and tested there.
 */
class TokenizersTest {

    @AfterEach
    void clearProperties() {
        System.getProperties().stringPropertyNames().stream()
                .filter(
                        key ->
                                key.startsWith("jinfer.preTokenizer.")
                                        || key.startsWith("toknroll.gguf.pre."))
                .toList()
                .forEach(System::clearProperty);
    }

    @Test
    void requireSameIdSpaceAcceptsAMatchingVocabulary() {
        GGUF gguf = ggufWithTokens("a", "b", "c");
        Tokenizer tokenizer = tokenizerWithVocabularyOf(3);
        assertDoesNotThrow(() -> Tokenizers.requireSameIdSpace(gguf, tokenizer));
    }

    @Test
    void requireSameIdSpaceRefusesAMismatchedVocabulary() {
        GGUF gguf = ggufWithTokens("a", "b", "c");
        Tokenizer tokenizer = tokenizerWithVocabularyOf(2);
        IllegalArgumentException e =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> Tokenizers.requireSameIdSpace(gguf, tokenizer));
        assertTrue(e.getMessage().contains("embedding table"), e.getMessage());
    }

    @Test
    void requireSameIdSpaceSkipsHeadersWithoutAVocabulary() {
        GGUF gguf = Builder.newBuilder().putString("general.name", "x").build();
        assertDoesNotThrow(() -> Tokenizers.requireSameIdSpace(gguf, tokenizerWithVocabularyOf(1)));
    }

    @Test
    void hasPropertyOverridesSeesBothNamespaces() {
        assertFalse(Tokenizers.hasPropertyOverrides());
        System.setProperty("toknroll.gguf.pre.my-scheme", "regex:.");
        assertTrue(Tokenizers.hasPropertyOverrides());
        System.clearProperty("toknroll.gguf.pre.my-scheme");
        System.setProperty("jinfer.preTokenizer.my-scheme", "alias:llama-bpe");
        assertTrue(Tokenizers.hasPropertyOverrides());
    }

    @Test
    void theMovedFlagIsRefusedWithItsNewName() {
        System.setProperty("jinfer.preTokenizer.my-scheme", "alias:llama-bpe");
        IllegalArgumentException e =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> Tokenizers.fromGGUF(Builder.newBuilder().build()));
        assertTrue(e.getMessage().contains("toknroll.gguf.pre.my-scheme"), e.getMessage());
        assertEquals("alias:llama-bpe", System.getProperty("jinfer.preTokenizer.my-scheme"));
    }

    private static GGUF ggufWithTokens(String... tokens) {
        return Builder.newBuilder().putArrayOfString("tokenizer.ggml.tokens", tokens).build();
    }

    /** Only the vocabulary size matters to the guard, so a pipeline-free stub suffices. */
    private static Tokenizer tokenizerWithVocabularyOf(int size) {
        String[] tokens = new String[size];
        for (int i = 0; i < size; i++) {
            tokens[i] = "t" + i;
        }
        var vocabulary = Toknroll.vocabulary(tokens);
        return new Tokenizer() {
            @Override
            public com.qxotic.toknroll.Vocabulary vocabulary() {
                return vocabulary;
            }

            @Override
            public void encodeInto(
                    CharSequence text,
                    int startInclusive,
                    int endExclusive,
                    com.qxotic.toknroll.IntSequence.Builder out) {
                throw new UnsupportedOperationException();
            }

            @Override
            public int decodeBytesInto(
                    com.qxotic.toknroll.IntSequence tokens,
                    int tokenStartIndex,
                    java.nio.ByteBuffer out) {
                throw new UnsupportedOperationException();
            }

            @Override
            public int countTokens(CharSequence text, int startInclusive, int endExclusive) {
                throw new UnsupportedOperationException();
            }
        };
    }
}
