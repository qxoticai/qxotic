package com.qxotic.toknroll;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import org.junit.jupiter.api.Test;

class TokenizerDefaultsTest {

    @Test
    void encodeIntoBuilderDelegatesToSliceOverload() {
        FakeTokenizer tokenizer = new FakeTokenizer();
        IntSequence.Builder out = IntSequence.newBuilder();

        tokenizer.encodeInto("abcd", out);

        assertEquals(0, tokenizer.lastStart);
        assertEquals(4, tokenizer.lastEnd);
        assertArrayEquals(new int[] {97, 98, 99, 100}, out.build().toArray());
    }

    @Test
    void encodeAndEncodeToArrayMatchSliceEncoding() {
        FakeTokenizer tokenizer = new FakeTokenizer();
        IntSequence encoded = tokenizer.encode("Az");

        assertArrayEquals(new int[] {65, 122}, encoded.toArray());
        assertArrayEquals(encoded.toArray(), tokenizer.encodeToArray("Az"));
    }

    @Test
    void decodeBytesUsesCountBytesAndDecodeInto() {
        FakeTokenizer tokenizer = new FakeTokenizer();
        IntSequence tokens = IntSequence.of(65, 66, 67);

        assertArrayEquals("ABC".getBytes(StandardCharsets.UTF_8), tokenizer.decodeBytes(tokens));
    }

    @Test
    void countTokensMatchesEncodeLengthInFake() {
        FakeTokenizer tokenizer = new FakeTokenizer();
        assertEquals(5, tokenizer.countTokens("hello"));
    }

    @Test
    void decodeIntArrayAndIntSequenceAreEquivalent() {
        FakeTokenizer tokenizer = new FakeTokenizer();
        int[] ids = {65, 66, 67};

        assertEquals("ABC", tokenizer.decode(ids));
        assertArrayEquals("ABC".getBytes(StandardCharsets.UTF_8), tokenizer.decodeBytes(ids));
    }

    @Test
    void decodeBytesIntoThrowsWhenNoTokenFits() {
        FakeTokenizer tokenizer = new FakeTokenizer();
        IntSequence tokens = IntSequence.of(65);
        ByteBuffer out = ByteBuffer.allocate(0);

        assertThrows(
                IllegalArgumentException.class, () -> tokenizer.decodeBytesInto(tokens, 0, out));
    }

    private static class FakeTokenizer implements Tokenizer {
        int lastStart = -1;
        int lastEnd = -1;

        @Override
        public Vocabulary vocabulary() {
            return null;
        }

        @Override
        public void encodeInto(
                CharSequence text, int startInclusive, int endExclusive, IntSequence.Builder out) {
            lastStart = startInclusive;
            lastEnd = endExclusive;
            for (int i = startInclusive; i < endExclusive; i++) {
                out.add(text.charAt(i));
            }
        }

        @Override
        public int decodeBytesInto(IntSequence tokens, int tokenStartIndex, ByteBuffer out) {
            if (tokenStartIndex < 0 || tokenStartIndex > tokens.length()) {
                throw new IndexOutOfBoundsException("tokenStartIndex: " + tokenStartIndex);
            }
            if (tokenStartIndex == tokens.length()) {
                return 0;
            }
            int consumed = 0;
            for (int i = tokenStartIndex; i < tokens.length(); i++) {
                if (out.remaining() == 0) {
                    if (consumed == 0) {
                        throw new IllegalArgumentException("Not enough output space");
                    }
                    break;
                }
                out.put((byte) tokens.intAt(i));
                consumed++;
            }
            return consumed;
        }

        @Override
        public int countTokens(CharSequence text) {
            return text.length();
        }

        @Override
        public int countBytes(IntSequence tokens) {
            return tokens.length();
        }
    }
}
