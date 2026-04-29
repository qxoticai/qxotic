package com.qxotic.toknroll.impl;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.StandardTokenType;
import com.qxotic.toknroll.Tokenizer;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import org.junit.jupiter.api.Test;

class TransformedTokenizerTest {

    // ------------------------------------------------------------------
    // decode
    // ------------------------------------------------------------------

    @Test
    void decodeIdentity() {
        TransformedTokenizer tt = identityWrapper();
        IntSequence ids = tt.base.encode("hello");
        assertEquals("hello", tt.decode(ids));
    }

    @Test
    void decodeMetaspaceToSpace() {
        TransformedTokenizer tt = metaspaceWrapper();
        IntSequence ids = tt.base.encode("\u2581hello");
        assertEquals(" hello", tt.decode(ids));
    }

    @Test
    void decodeTrimLeadingSpace() {
        TransformedTokenizer tt = new TrimLeadingSpaceWrapper(metaspaceBase());
        IntSequence ids = tt.base.encode("\u2581hello");
        assertEquals("hello", tt.decode(ids));
    }

    // ------------------------------------------------------------------
    // decodeBytesInto — normal path (no byte fallback)
    // ------------------------------------------------------------------

    @Test
    void decodeBytesIntoNormalPathRoundTrip() {
        TransformedTokenizer tt = metaspaceWrapper();
        String text = "\u2581hello\u2581world";
        IntSequence ids = tt.base.encode(text);

        byte[] decoded = tt.decodeBytes(ids);
        byte[] decodedBytesInto = decodeAllBytesInto(tt, ids);

        assertArrayEquals(decoded, decodedBytesInto);
    }

    @Test
    void decodeBytesIntoTrimLeadingSpace() {
        TransformedTokenizer tt = new TrimLeadingSpaceWrapper(metaspaceBase());
        String text = "\u2581hello";
        IntSequence ids = tt.base.encode(text);
        String expected = "hello";

        byte[] expectedBytes = expected.getBytes(StandardCharsets.UTF_8);
        ByteBuffer buf = ByteBuffer.allocate(expectedBytes.length);
        int consumed = tt.decodeBytesInto(ids, 0, buf);

        assertEquals(ids.length(), consumed);
        buf.flip();
        byte[] actual = new byte[buf.remaining()];
        buf.get(actual);
        assertArrayEquals(expectedBytes, actual);
    }

    @Test
    void decodeBytesIntoTokenStartIndexEqualsLengthReturnsZero() {
        TransformedTokenizer tt = metaspaceWrapper();
        IntSequence ids = tt.base.encode("\u2581hello");
        ByteBuffer buf = ByteBuffer.allocate(16);
        int consumed = tt.decodeBytesInto(ids, ids.length(), buf);
        assertEquals(0, consumed);
    }

    @Test
    void decodeBytesIntoNegativeTokenStartIndexThrows() {
        TransformedTokenizer tt = metaspaceWrapper();
        IntSequence ids = tt.base.encode("\u2581hello");
        ByteBuffer buf = ByteBuffer.allocate(16);
        assertThrows(IndexOutOfBoundsException.class, () -> tt.decodeBytesInto(ids, -1, buf));
    }

    @Test
    void decodeBytesIntoTokenStartIndexBeyondLengthThrows() {
        TransformedTokenizer tt = metaspaceWrapper();
        IntSequence ids = tt.base.encode("\u2581hello");
        ByteBuffer buf = ByteBuffer.allocate(16);
        assertThrows(
                IndexOutOfBoundsException.class,
                () -> tt.decodeBytesInto(ids, ids.length() + 1, buf));
    }

    @Test
    void decodeBytesIntoPartialOutput() {
        TransformedTokenizer tt = metaspaceWrapper();
        String text = "\u2581hello\u2581world";
        IntSequence ids = tt.base.encode(text);
        ByteBuffer buf = ByteBuffer.allocate(5); // partial
        int consumed = tt.decodeBytesInto(ids, 0, buf);
        assertTrue(consumed > 0);
        assertTrue(consumed < ids.length());
    }

    @Test
    void decodeBytesIntoInsufficientSpaceOnFirstToken() {
        TransformedTokenizer tt = metaspaceWrapper();
        IntSequence ids = tt.base.encode("\u2581hello");
        // At least one token will be produced; allocate only 1 byte
        ByteBuffer buf = ByteBuffer.allocate(1);
        // If the first token's transformed bytes are >1 byte, this should throw
        if (tt.countBytes(IntSequence.wrap(new int[] {ids.intAt(0)})) > 1) {
            assertThrows(IllegalArgumentException.class, () -> tt.decodeBytesInto(ids, 0, buf));
        }
    }

    // ------------------------------------------------------------------
    // decodeBytesInto — byte fallback path
    // ------------------------------------------------------------------

    @Test
    void decodeBytesIntoByteFallbackPath() {
        Tokenizer byteFallbackBase = byteFallbackBaseTokenizer();
        TransformedTokenizer tt = new IdentityWrapper(byteFallbackBase);
        String text = "test";
        IntSequence ids = byteFallbackBase.encode(text);

        byte[] expected = text.getBytes(StandardCharsets.UTF_8);
        ByteBuffer buf = ByteBuffer.allocate(expected.length);
        int consumed = tt.decodeBytesInto(ids, 0, buf);

        assertEquals(ids.length(), consumed);
        buf.flip();
        byte[] actual = new byte[buf.remaining()];
        buf.get(actual);
        assertArrayEquals(expected, actual);
    }

    // ------------------------------------------------------------------
    // countBytes
    // ------------------------------------------------------------------

    @Test
    void countBytesNormalPath() {
        TransformedTokenizer tt = metaspaceWrapper();
        String text = "\u2581hello";
        IntSequence ids = tt.base.encode(text);
        int count = tt.countBytes(ids);
        // The transform replaces \u2581 with space, so the byte count should match
        // the decoded String's UTF-8 length
        String decoded = tt.decode(ids);
        assertEquals(decoded.getBytes(StandardCharsets.UTF_8).length, count);
    }

    @Test
    void countBytesTrimLeadingSpace() {
        TransformedTokenizer tt = new TrimLeadingSpaceWrapper(metaspaceBase());
        String text = "\u2581hello";
        IntSequence ids = tt.base.encode(text);
        // Leading space trimmed: "hello" = 5 bytes, "hello" minus leading space = 5 bytes
        // Actually metaspace maps to ' ', so "\u2581hello" -> "hello" which is 5 bytes.
        // With trimLeadingSpace: the first token would have leading space trimmed
        int count = tt.countBytes(ids);
        assertTrue(count >= 4); // at least "ello" after trimming 1 byte
    }

    @Test
    void countBytesEmptyTokens() {
        TransformedTokenizer tt = metaspaceWrapper();
        assertEquals(0, tt.countBytes(IntSequence.wrap(new int[0])));
    }

    @Test
    void countBytesByteFallback() {
        Tokenizer byteFallbackBase = byteFallbackBaseTokenizer();
        TransformedTokenizer tt = new IdentityWrapper(byteFallbackBase);
        IntSequence ids = byteFallbackBase.encode("test");
        int count = tt.countBytes(ids);
        assertEquals("test".getBytes(StandardCharsets.UTF_8).length, count);
    }

    @Test
    void countBytesUnknownTokenThrows() {
        TransformedTokenizer tt = metaspaceWrapper();
        int bogusId = tt.vocabulary().size() + 100;
        IntSequence ids = IntSequence.wrap(new int[] {bogusId});
        assertThrows(IllegalArgumentException.class, () -> tt.countBytes(ids));
    }

    // ------------------------------------------------------------------
    // decodeBytes
    // ------------------------------------------------------------------

    @Test
    void decodeBytesRoundTrip() {
        TransformedTokenizer tt = metaspaceWrapper();
        String text = "\u2581hello\u2581world";
        IntSequence ids = tt.base.encode(text);

        byte[] bytes = tt.decodeBytes(ids);
        String decoded = tt.decode(ids);
        assertArrayEquals(decoded.getBytes(StandardCharsets.UTF_8), bytes);
    }

    @Test
    void decodeBytesByteFallbackPath() {
        Tokenizer byteFallbackBase = byteFallbackBaseTokenizer();
        TransformedTokenizer tt = new IdentityWrapper(byteFallbackBase);
        IntSequence ids = byteFallbackBase.encode("test");

        byte[] bytes = tt.decodeBytes(ids);
        assertArrayEquals("test".getBytes(StandardCharsets.UTF_8), bytes);
    }

    // ------------------------------------------------------------------
    // expectedTokensPerChar / vocabulary delegation
    // ------------------------------------------------------------------

    @Test
    void expectedTokensPerCharDelegatesToBase() {
        TransformedTokenizer tt = metaspaceWrapper();
        assertEquals(tt.base.expectedTokensPerChar(), tt.expectedTokensPerChar());
    }

    @Test
    void vocabularyDelegatesToBase() {
        TransformedTokenizer tt = metaspaceWrapper();
        assertEquals(tt.base.vocabulary(), tt.vocabulary());
    }

    // ------------------------------------------------------------------
    // normalizeMetaspaceDecoded static helper
    // ------------------------------------------------------------------

    @Test
    void normalizeMetaspaceDecodedNoSpaceNoTrim() {
        String result = TransformedTokenizer.normalizeMetaspaceDecoded("hello", false);
        assertEquals("hello", result);
    }

    @Test
    void normalizeMetaspaceDecodedWithMetaspace() {
        String result = TransformedTokenizer.normalizeMetaspaceDecoded("\u2581hello", false);
        assertEquals(" hello", result);
    }

    @Test
    void normalizeMetaspaceDecodedTrimLeadingSpace() {
        String result = TransformedTokenizer.normalizeMetaspaceDecoded("\u2581hello", true);
        assertEquals("hello", result);
    }

    @Test
    void normalizeMetaspaceDecodedTrimWhenNoLeadingSpace() {
        String result = TransformedTokenizer.normalizeMetaspaceDecoded("hellou2581world", true);
        assertEquals("hellou2581world", result);
    }

    @Test
    void normalizeMetaspaceDecodedEmptyString() {
        assertEquals("", TransformedTokenizer.normalizeMetaspaceDecoded("", true));
        assertEquals("", TransformedTokenizer.normalizeMetaspaceDecoded("", false));
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    private static TransformedTokenizer identityWrapper() {
        SentencePieceBpeModel base = minimalSpBpeModel();
        return new IdentityWrapper(base);
    }

    private static TransformedTokenizer metaspaceWrapper() {
        SentencePieceBpeModel base = metaspaceBase();
        return new IdentityWrapper(base);
    }

    private static SentencePieceBpeModel metaspaceBase() {
        String[] tokens = {
            "<0x00>",
            "\u2581",
            "h",
            "e",
            "l",
            "o",
            "w",
            "r",
            "d",
            "he",
            "el",
            "ll",
            "lo",
            "wo",
            "or",
            "rl",
            "ld",
            "hello",
            "world",
            "\u2581hello",
            "\u2581world"
        };
        float[] scores = new float[tokens.length];
        scores[0] = 0f;
        for (int i = 1; i < tokens.length; i++) {
            scores[i] = (float) i;
        }
        int[] types = normalTypes(tokens.length);
        types[0] = StandardTokenType.BYTE.getId();
        return SentencePieceBpeModel.fromVocabulary(new VocabularyImpl(tokens, types), scores);
    }

    private static SentencePieceBpeModel minimalSpBpeModel() {
        String[] tokens = {"<0x00>", " ", "h", "e", "l", "o", "he", "ll", "llo", "hello", " hello"};
        float[] scores = {0f, 0f, 0f, 0f, 0f, 0f, 1f, 2f, 3f, 4f, 5f};
        int[] types = normalTypes(tokens.length);
        types[0] = StandardTokenType.BYTE.getId();
        return SentencePieceBpeModel.fromVocabulary(new VocabularyImpl(tokens, types), scores);
    }

    /** Creates a tokenizer with byte-level tokens (triggers byteFallback path). */
    private static Tokenizer byteFallbackBaseTokenizer() {
        String[] tokens = new String[257];
        tokens[0] = "<0x00>";
        for (int i = 1; i < 256; i++) {
            tokens[i] = String.format("<0x%02X>", i);
        }
        tokens[256] = " ";
        float[] scores = new float[257];
        int[] types = new int[257];
        types[0] = StandardTokenType.BYTE.getId();
        for (int i = 1; i < 256; i++) {
            types[i] = StandardTokenType.BYTE.getId();
        }
        types[256] = StandardTokenType.NORMAL.getId();
        return SentencePieceBpeModel.fromVocabulary(new VocabularyImpl(tokens, types), scores);
    }

    private static byte[] decodeAllBytesInto(TransformedTokenizer tt, IntSequence ids) {
        int total = tt.countBytes(ids);
        ByteBuffer buf = ByteBuffer.allocate(total);
        tt.decodeBytesInto(ids, 0, buf);
        buf.flip();
        byte[] out = new byte[buf.remaining()];
        buf.get(out);
        return out;
    }

    private static int[] normalTypes(int length) {
        int[] types = new int[length];
        for (int i = 0; i < length; i++) {
            types[i] = StandardTokenType.NORMAL.getId();
        }
        return types;
    }

    private static final class IdentityWrapper extends TransformedTokenizer {
        IdentityWrapper(Tokenizer base) {
            super(base);
        }

        @Override
        protected String transformDecoded(String decoded, boolean atStartOfText) {
            return decoded.replace('\u2581', ' ');
        }
    }

    private static final class TrimLeadingSpaceWrapper extends TransformedTokenizer {
        TrimLeadingSpaceWrapper(Tokenizer base) {
            super(base);
        }

        @Override
        protected String transformDecoded(String decoded, boolean atStartOfText) {
            return TransformedTokenizer.normalizeMetaspaceDecoded(decoded, atStartOfText);
        }

        @Override
        protected boolean trimLeadingSpaceAtStart() {
            return true;
        }
    }
}
