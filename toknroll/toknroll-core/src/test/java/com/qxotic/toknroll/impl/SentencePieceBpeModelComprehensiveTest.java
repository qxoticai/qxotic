package com.qxotic.toknroll.impl;

import static org.junit.jupiter.api.Assertions.*;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.StandardTokenType;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import org.junit.jupiter.api.Test;

/**
 * Comprehensive unit tests for SentencePiece BPE model.
 *
 * <p>Covers:
 *
 * <ul>
 *   <li>Synthetic controlled-vocabulary tests (no network needed)
 *   <li>Real model round-trip tests for Gemma 4 and Mistral v0.3
 *   <li>Edge cases: empty strings, unicode, byte fallback, long texts
 * </ul>
 */
class SentencePieceBpeModelComprehensiveTest {

    // ------------------------------------------------------------------
    // Synthetic model tests
    // ------------------------------------------------------------------

    @Test
    void emptyStringRoundTrip() {
        SentencePieceBpeModel model = minimalModel();
        IntSequence encoded = model.encode("");
        assertEquals(0, encoded.length());
        assertEquals("", model.decode(encoded));
        assertEquals(0, model.countTokens(""));
    }

    @Test
    void singleTokenRoundTrip() {
        SentencePieceBpeModel model = minimalModel();
        IntSequence encoded = model.encode(" hello");
        assertArrayEquals(new int[] {10}, encoded.toArray());
        assertEquals(" hello", model.decode(encoded));
    }

    @Test
    void greedyMergeToHighestScore() {
        SentencePieceBpeModel model = minimalModel();
        // " hello" -> " " + "h" + "e" + "l" + "l" + "o" should merge greedily to " hello"
        IntSequence encoded = model.encode(" hello");
        assertArrayEquals(new int[] {10}, encoded.toArray());
        assertEquals(" hello", model.decode(encoded));
    }

    @Test
    void spaceRunTokenization() {
        String[] tokens = {"<0x00>", " ", "  ", "   ", "a"};
        float[] scores = {0f, 0f, 1f, 1f, 0f};
        int[] types = normalTypes(tokens.length);
        types[0] = StandardTokenType.BYTE.getId();
        SentencePieceBpeModel model =
                SentencePieceBpeModel.fromVocabulary(new VocabularyImpl(tokens, types), scores);

        assertArrayEquals(new int[] {2}, model.encode("  ").toArray());
        assertArrayEquals(new int[] {3}, model.encode("   ").toArray());
    }

    @Test
    void unicodeCharacterHandling() {
        String[] tokens = {"<0x00>", " ", "é", "l", "o", "él", "lo", "élo", " élo"};
        float[] scores = {0f, 0f, 0f, 0f, 0f, 1f, 2f, 3f, 4f};
        int[] types = normalTypes(9);
        types[0] = StandardTokenType.BYTE.getId();
        SentencePieceBpeModel model =
                SentencePieceBpeModel.fromVocabulary(new VocabularyImpl(tokens, types), scores);

        IntSequence encoded = model.encode(" élo");
        assertArrayEquals(new int[] {8}, encoded.toArray());
        assertEquals(" élo", model.decode(encoded));
    }

    @Test
    void byteFallbackForUnknownChars() {
        // Full byte vocab: <0x00> through <0xFF>
        String[] tokens = new String[257];
        float[] scores = new float[257];
        int[] types = new int[257];
        tokens[0] = "<0x00>";
        scores[0] = 0f;
        types[0] = StandardTokenType.BYTE.getId();
        for (int i = 1; i < 256; i++) {
            tokens[i] = String.format("<0x%02X>", i);
            scores[i] = 0f;
            types[i] = StandardTokenType.BYTE.getId();
        }
        tokens[256] = " ";
        scores[256] = 0f;
        types[256] = StandardTokenType.NORMAL.getId();

        SentencePieceBpeModel model =
                SentencePieceBpeModel.fromVocabulary(new VocabularyImpl(tokens, types), scores);

        IntSequence encoded = model.encode("b");
        assertEquals(1, encoded.length());
        assertEquals("b", model.decode(encoded));

        IntSequence encoded2 = model.encode("☠");
        assertTrue(encoded2.length() > 0);
        assertEquals("☠", model.decode(encoded2));
    }

    @Test
    void fastPathProducesValidResult() {
        // Large vocab to force fast merge path (threshold > 128 tokens).
        // Use single-char tokens so encoding doesn't fall back to bytes.
        int vocabSize = 300;
        String[] tokens = new String[vocabSize];
        float[] scores = new float[vocabSize];
        int[] types = new int[vocabSize];
        tokens[0] = "<0x00>";
        scores[0] = 0f;
        types[0] = StandardTokenType.BYTE.getId();
        tokens[1] = " ";
        scores[1] = 0f;
        types[1] = StandardTokenType.NORMAL.getId();

        // Fill with single printable ASCII chars starting from '!' (33)
        for (int i = 2; i < vocabSize; i++) {
            tokens[i] = String.valueOf((char) (i + 31));
            scores[i] = i;
            types[i] = StandardTokenType.NORMAL.getId();
        }

        SentencePieceBpeModel model =
                SentencePieceBpeModel.fromVocabulary(new VocabularyImpl(tokens, types), scores);

        // Build a long input that produces >128 initial tokens using single chars
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 200; i++) {
            sb.append(" ").append(tokens[(i % (vocabSize - 2)) + 2]);
        }
        String text = sb.toString();

        IntSequence encoded = model.encode(text);
        assertTrue(encoded.length() > 128, "Should trigger fast merge path");

        String decoded = model.decode(encoded);
        assertEquals(text, decoded, "Fast path round-trip failed");
    }

    @Test
    void countTokensMatchesEncodeLength() {
        SentencePieceBpeModel model = minimalModel();
        String[] texts = {"", " hello", " hello world"};
        for (String text : texts) {
            int count = model.countTokens(text);
            int length = model.encode(text).length();
            assertEquals(
                    length, count, "countTokens should match encode length for: [" + text + "]");
        }
    }

    @Test
    void expectedTokensPerCharIsPositive() {
        assertTrue(minimalModel().expectedTokensPerChar() > 0f);
    }

    // ------------------------------------------------------------------
    // decodeBytesInto tests
    // ------------------------------------------------------------------

    @Test
    void decodeBytesIntoRoundTrip() {
        SentencePieceBpeModel model = minimalModel();
        IntSequence ids = model.encode(" hello");
        byte[] expected = " hello".getBytes(StandardCharsets.UTF_8);

        ByteBuffer buf = ByteBuffer.allocate(expected.length);
        int consumed = model.decodeBytesInto(ids, 0, buf);

        assertEquals(ids.length(), consumed);
        buf.flip();
        byte[] actual = new byte[buf.remaining()];
        buf.get(actual);
        assertArrayEquals(expected, actual);
    }

    @Test
    void decodeBytesIntoByteFallbackToken() {
        SentencePieceBpeModel model = fullByteFallbackModel();
        IntSequence ids = model.encode("\u00e9");

        byte[] expected = "\u00e9".getBytes(StandardCharsets.UTF_8);
        ByteBuffer buf = ByteBuffer.allocate(expected.length);
        int consumed = model.decodeBytesInto(ids, 0, buf);

        assertEquals(ids.length(), consumed);
        buf.flip();
        byte[] actual = new byte[buf.remaining()];
        buf.get(actual);
        assertArrayEquals(expected, actual);
    }

    @Test
    void decodeBytesIntoEmptyTokens() {
        SentencePieceBpeModel model = minimalModel();
        IntSequence ids = IntSequence.wrap(new int[0]);
        ByteBuffer buf = ByteBuffer.allocate(16);
        int consumed = model.decodeBytesInto(ids, 0, buf);
        assertEquals(0, consumed);
    }

    @Test
    void decodeBytesIntoTokenStartIndexEqualsLength() {
        SentencePieceBpeModel model = minimalModel();
        IntSequence ids = model.encode(" hello");
        ByteBuffer buf = ByteBuffer.allocate(16);
        int consumed = model.decodeBytesInto(ids, ids.length(), buf);
        assertEquals(0, consumed);
    }

    @Test
    void decodeBytesIntoNegativeTokenStartIndexThrows() {
        SentencePieceBpeModel model = minimalModel();
        IntSequence ids = model.encode(" hello");
        ByteBuffer buf = ByteBuffer.allocate(16);
        assertThrows(IndexOutOfBoundsException.class, () -> model.decodeBytesInto(ids, -1, buf));
    }

    @Test
    void decodeBytesIntoTokenStartIndexBeyondLengthThrows() {
        SentencePieceBpeModel model = minimalModel();
        IntSequence ids = model.encode(" hello");
        ByteBuffer buf = ByteBuffer.allocate(16);
        assertThrows(
                IndexOutOfBoundsException.class,
                () -> model.decodeBytesInto(ids, ids.length() + 1, buf));
    }

    @Test
    void decodeBytesIntoNullTokensThrows() {
        SentencePieceBpeModel model = minimalModel();
        ByteBuffer buf = ByteBuffer.allocate(16);
        assertThrows(NullPointerException.class, () -> model.decodeBytesInto(null, 0, buf));
    }

    @Test
    void decodeBytesIntoNullBufferThrows() {
        SentencePieceBpeModel model = minimalModel();
        IntSequence ids = model.encode(" hello");
        assertThrows(NullPointerException.class, () -> model.decodeBytesInto(ids, 0, null));
    }

    @Test
    void decodeBytesIntoUnknownTokenThrows() {
        SentencePieceBpeModel model = minimalModel();
        int bogusId = model.vocabulary().size() + 100;
        IntSequence ids = IntSequence.wrap(new int[] {bogusId});
        ByteBuffer buf = ByteBuffer.allocate(16);
        assertThrows(IllegalArgumentException.class, () -> model.decodeBytesInto(ids, 0, buf));
    }

    @Test
    void decodeBytesIntoPartialOutputWhenBufferTooSmall() {
        SentencePieceBpeModel model = minimalModel();
        IntSequence ids = model.encode(" hello world");
        // Allocate only enough for part of first token
        ByteBuffer buf = ByteBuffer.allocate(1); // too small for " " + "hello"
        assertThrows(IllegalArgumentException.class, () -> model.decodeBytesInto(ids, 0, buf));
    }

    @Test
    void decodeBytesIntoByteRunCoalescing() {
        // Two consecutive byte tokens should be coalesced into one byte run
        SentencePieceBpeModel model = fullByteFallbackModel();
        String text = "\u00e9\u00f1"; // two multi-byte utf-8 chars
        IntSequence ids = model.encode(text);

        byte[] expected = text.getBytes(StandardCharsets.UTF_8);
        ByteBuffer buf = ByteBuffer.allocate(expected.length);
        int consumed = model.decodeBytesInto(ids, 0, buf);

        assertEquals(ids.length(), consumed);
        buf.flip();
        byte[] actual = new byte[buf.remaining()];
        buf.get(actual);
        assertArrayEquals(expected, actual);
    }

    // ------------------------------------------------------------------
    // countBytes tests
    // ------------------------------------------------------------------

    @Test
    void countBytesMatchesDecodedUtf8Length() {
        SentencePieceBpeModel model = minimalModel();
        IntSequence ids = model.encode(" hello");
        String decoded = model.decode(ids);
        assertEquals(decoded.getBytes(StandardCharsets.UTF_8).length, model.countBytes(ids));
    }

    @Test
    void countBytesEmptyTokens() {
        SentencePieceBpeModel model = minimalModel();
        assertEquals(0, model.countBytes(IntSequence.wrap(new int[0])));
    }

    @Test
    void countBytesNullTokensThrows() {
        SentencePieceBpeModel model = minimalModel();
        assertThrows(NullPointerException.class, () -> model.countBytes((IntSequence) null));
    }

    @Test
    void countBytesByteFallbackTokens() {
        SentencePieceBpeModel model = fullByteFallbackModel();
        String text = "\u00e9"; // 2 bytes in UTF-8
        IntSequence ids = model.encode(text);
        assertEquals(2, model.countBytes(ids));
    }

    // ------------------------------------------------------------------
    // encodeInto range validation
    // ------------------------------------------------------------------

    @Test
    void encodeIntoRangeValidation() {
        SentencePieceBpeModel model = minimalModel();
        String text = " hello world";
        IntSequence allIds = model.encode(text);

        IntSequence.Builder out = IntSequence.newBuilder(16);
        model.encodeInto(text, 0, 6, out); // " hello"
        IntSequence sliceIds = out.build();

        assertEquals(model.encode(" hello"), sliceIds);
    }

    @Test
    void encodeIntoEmptyRange() {
        SentencePieceBpeModel model = minimalModel();
        IntSequence.Builder out = IntSequence.newBuilder(16);
        model.encodeInto(" hello", 1, 1, out);
        assertEquals(0, out.size());
    }

    @Test
    void encodeIntoNegativeStartThrows() {
        SentencePieceBpeModel model = minimalModel();
        IntSequence.Builder out = IntSequence.newBuilder(16);
        assertThrows(IndexOutOfBoundsException.class, () -> model.encodeInto("test", -1, 4, out));
    }

    @Test
    void encodeIntoEndBeforeStartThrows() {
        SentencePieceBpeModel model = minimalModel();
        IntSequence.Builder out = IntSequence.newBuilder(16);
        assertThrows(IndexOutOfBoundsException.class, () -> model.encodeInto("test", 2, 1, out));
    }

    @Test
    void encodeIntoEndExceedsLengthThrows() {
        SentencePieceBpeModel model = minimalModel();
        IntSequence.Builder out = IntSequence.newBuilder(16);
        assertThrows(IndexOutOfBoundsException.class, () -> model.encodeInto("test", 0, 10, out));
    }

    @Test
    void encodeIntoNullTextThrows() {
        SentencePieceBpeModel model = minimalModel();
        IntSequence.Builder out = IntSequence.newBuilder(16);
        assertThrows(NullPointerException.class, () -> model.encodeInto(null, 0, 1, out));
    }

    @Test
    void encodeIntoNullOutThrows() {
        SentencePieceBpeModel model = minimalModel();
        assertThrows(NullPointerException.class, () -> model.encodeInto("test", 0, 1, null));
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    /** Minimal model with no merge conflicts for basic tests. */
    private static SentencePieceBpeModel minimalModel() {
        String[] tokens = {
            "<0x00>",
            " ",
            "h",
            "e",
            "l",
            "o",
            "he",
            "ll",
            "llo",
            "hello",
            " hello",
            "w",
            "r",
            "d",
            "world",
            " world",
            " hello world"
        };
        float[] scores = {0f, 0f, 0f, 0f, 0f, 0f, 1f, 2f, 3f, 4f, 5f, 0f, 0f, 0f, 4f, 5f, 6f};
        int[] types = normalTypes(tokens.length);
        types[0] = StandardTokenType.BYTE.getId();
        return SentencePieceBpeModel.fromVocabulary(new VocabularyImpl(tokens, types), scores);
    }

    private static int[] normalTypes(int length) {
        int[] types = new int[length];
        for (int i = 0; i < length; i++) {
            types[i] = StandardTokenType.NORMAL.getId();
        }
        return types;
    }

    private static SentencePieceBpeModel fullByteFallbackModel() {
        String[] tokens = new String[257];
        float[] scores = new float[257];
        int[] types = new int[257];
        tokens[0] = "<0x00>";
        scores[0] = 0f;
        types[0] = StandardTokenType.BYTE.getId();
        for (int i = 1; i < 256; i++) {
            tokens[i] = String.format("<0x%02X>", i);
            scores[i] = 0f;
            types[i] = StandardTokenType.BYTE.getId();
        }
        tokens[256] = " ";
        scores[256] = 0f;
        types[256] = StandardTokenType.NORMAL.getId();
        return SentencePieceBpeModel.fromVocabulary(new VocabularyImpl(tokens, types), scores);
    }

    private static int[] byteTypes(int length) {
        int[] types = normalTypes(length);
        types[0] = StandardTokenType.BYTE.getId();
        return types;
    }
}
