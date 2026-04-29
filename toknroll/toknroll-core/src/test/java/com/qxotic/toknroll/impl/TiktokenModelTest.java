package com.qxotic.toknroll.impl;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.toknroll.ByteLevel;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.StandardTokenType;
import com.qxotic.toknroll.Vocabulary;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.NoSuchElementException;
import org.junit.jupiter.api.Test;

class TiktokenModelTest {

    // ------------------------------------------------------------------
    // Encode / decode round-trip
    // ------------------------------------------------------------------

    @Test
    void encodeDecodeRoundTrip() {
        TiktokenModel model = model256();
        String text = "Hello, World!";
        IntSequence ids = model.encode(text);
        assertEquals(text, model.decode(ids));
    }

    @Test
    void encodeDecodeEmptyString() {
        TiktokenModel model = model256();
        IntSequence ids = model.encode("");
        assertEquals(0, ids.length());
        assertEquals("", model.decode(ids));
    }

    @Test
    void encodeSingleAsciiChar() {
        TiktokenModel model = model256();
        IntSequence ids = model.encode("A");
        assertEquals(1, ids.length());
        assertEquals("A", model.decode(ids));
    }

    @Test
    void encodeUnicodeText() {
        TiktokenModel model = model256();
        String text = "héllo café\u00a0wörld";
        IntSequence ids = model.encode(text);
        assertEquals(text, model.decode(ids));
    }

    @Test
    void encodeWithSurrogatePairs() {
        TiktokenModel model = model256();
        String text = "emoji \ud83d\ude00 grin";
        IntSequence ids = model.encode(text);
        assertEquals(text, model.decode(ids));
    }

    // ------------------------------------------------------------------
    // encodeInto range validation
    // ------------------------------------------------------------------

    @Test
    void encodeIntoRangeRoundTrip() {
        TiktokenModel model = model256();
        String text = "abcdefghij";
        IntSequence idsAll = model.encode(text);

        IntSequence.Builder out = IntSequence.newBuilder(16);
        model.encodeInto(text, 1, 4, out);
        IntSequence idsSlice = out.build();

        assertEquals(text.substring(1, 4), model.decode(idsSlice));
    }

    @Test
    void encodeIntoEmptyRangeReturnsZeroTokens() {
        TiktokenModel model = model256();
        IntSequence.Builder out = IntSequence.newBuilder(16);
        model.encodeInto("abc", 1, 1, out);
        assertEquals(0, out.size());
    }

    @Test
    void encodeIntoStartNegativeThrows() {
        TiktokenModel model = model256();
        IntSequence.Builder out = IntSequence.newBuilder(16);
        assertThrows(IndexOutOfBoundsException.class, () -> model.encodeInto("abc", -1, 3, out));
    }

    @Test
    void encodeIntoEndBeforeStartThrows() {
        TiktokenModel model = model256();
        IntSequence.Builder out = IntSequence.newBuilder(16);
        assertThrows(IndexOutOfBoundsException.class, () -> model.encodeInto("abc", 2, 1, out));
    }

    @Test
    void encodeIntoEndExceedsLengthThrows() {
        TiktokenModel model = model256();
        IntSequence.Builder out = IntSequence.newBuilder(16);
        assertThrows(IndexOutOfBoundsException.class, () -> model.encodeInto("abc", 0, 10, out));
    }

    @Test
    void encodeIntoNullTextThrows() {
        TiktokenModel model = model256();
        IntSequence.Builder out = IntSequence.newBuilder(16);
        assertThrows(NullPointerException.class, () -> model.encodeInto(null, 0, 1, out));
    }

    @Test
    void encodeIntoNullOutThrows() {
        TiktokenModel model = model256();
        assertThrows(NullPointerException.class, () -> model.encodeInto("abc", 0, 1, null));
    }

    // ------------------------------------------------------------------
    // countTokens
    // ------------------------------------------------------------------

    @Test
    void countTokensMatchesEncodeLength() {
        TiktokenModel model = model256();
        String[] texts = {"", "a", "hello", "héllo wörld"};
        for (String text : texts) {
            assertEquals(
                    model.encode(text).length(),
                    model.countTokens(text),
                    "countTokens mismatch for: " + text);
        }
    }

    @Test
    void countTokensRanged() {
        TiktokenModel model = model256();
        String text = "abcdef";
        int count = model.countTokens(text, 1, 4);
        assertEquals(model.encode("bcd").length(), count);
    }

    @Test
    void countTokensNullTextThrows() {
        TiktokenModel model = model256();
        assertThrows(NullPointerException.class, () -> model.countTokens(null));
    }

    // ------------------------------------------------------------------
    // decodeBytesInto
    // ------------------------------------------------------------------

    @Test
    void decodeBytesIntoRoundTrip() {
        TiktokenModel model = model256();
        String text = "Hello";
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

    @Test
    void decodeBytesIntoWithTokenStartIndex() {
        TiktokenModel model = model256();
        IntSequence ids = model.encode("HelloWorld");

        ByteBuffer buf = ByteBuffer.allocate(16);
        int consumed = model.decodeBytesInto(ids, ids.length() / 2, buf);

        assertTrue(consumed > 0);
        assertTrue(consumed < ids.length());
    }

    @Test
    void decodeBytesIntoTokenStartIndexEqualsLengthReturnsZero() {
        TiktokenModel model = model256();
        IntSequence ids = model.encode("abc");
        ByteBuffer buf = ByteBuffer.allocate(16);
        int consumed = model.decodeBytesInto(ids, ids.length(), buf);
        assertEquals(0, consumed);
    }

    @Test
    void decodeBytesIntoNegativeTokenStartIndexThrows() {
        TiktokenModel model = model256();
        IntSequence ids = model.encode("abc");
        ByteBuffer buf = ByteBuffer.allocate(16);
        assertThrows(IndexOutOfBoundsException.class, () -> model.decodeBytesInto(ids, -1, buf));
    }

    @Test
    void decodeBytesIntoTokenStartIndexBeyondLengthThrows() {
        TiktokenModel model = model256();
        IntSequence ids = model.encode("abc");
        ByteBuffer buf = ByteBuffer.allocate(16);
        assertThrows(
                IndexOutOfBoundsException.class,
                () -> model.decodeBytesInto(ids, ids.length() + 1, buf));
    }

    @Test
    void decodeBytesIntoNullTokensThrows() {
        TiktokenModel model = model256();
        ByteBuffer buf = ByteBuffer.allocate(16);
        assertThrows(NullPointerException.class, () -> model.decodeBytesInto(null, 0, buf));
    }

    @Test
    void decodeBytesIntoNullBufferThrows() {
        TiktokenModel model = model256();
        IntSequence ids = model.encode("a");
        assertThrows(NullPointerException.class, () -> model.decodeBytesInto(ids, 0, null));
    }

    @Test
    void decodeBytesIntoUnknownTokenThrows() {
        TiktokenModel model = model256();
        int bogusId = model.vocabulary().size() + 100;
        IntSequence ids = IntSequence.wrap(new int[] {bogusId});
        ByteBuffer buf = ByteBuffer.allocate(16);
        assertThrows(NoSuchElementException.class, () -> model.decodeBytesInto(ids, 0, buf));
    }

    @Test
    void decodeBytesIntoInsufficientOutputSpaceOnFirstTokenThrows() {
        // Create a model with a multi-byte "hello" token. With ignoreMerges=true,
        // "hello" encodes as one 5-byte token, which won't fit in a 1-byte buffer.
        byte[] helloBytes = "hello".getBytes(StandardCharsets.UTF_8);
        String helloToken = ByteLevel.encode(helloBytes);
        List<String> extraTokens = new ArrayList<>();
        extraTokens.add(helloToken);
        TiktokenModel model = model256WithIgnoreMerges(extraTokens);

        IntSequence ids = model.encode("hello");
        assertEquals(1, ids.length());
        ByteBuffer buf = ByteBuffer.allocate(1);
        assertThrows(IllegalArgumentException.class, () -> model.decodeBytesInto(ids, 0, buf));
    }

    @Test
    void decodeBytesIntoPartialDecodeWhenOutputFull() {
        TiktokenModel model = model256();
        String text = "HelloWorld";
        IntSequence ids = model.encode(text);
        ByteBuffer buf = ByteBuffer.allocate(6); // partial fit
        int consumed = model.decodeBytesInto(ids, 0, buf);
        assertTrue(consumed > 0);
        assertTrue(consumed < ids.length());
    }

    // ------------------------------------------------------------------
    // countBytes
    // ------------------------------------------------------------------

    @Test
    void countBytesMatchesUtf8Length() {
        TiktokenModel model = model256();
        String text = "Hello, World!";
        IntSequence ids = model.encode(text);
        int byteLength = text.getBytes(StandardCharsets.UTF_8).length;
        assertEquals(byteLength, model.countBytes(ids));
    }

    @Test
    void countBytesEmptyTokens() {
        TiktokenModel model = model256();
        assertEquals(0, model.countBytes(IntSequence.wrap(new int[0])));
    }

    @Test
    void countBytesUnknownTokenThrows() {
        TiktokenModel model = model256();
        int bogusId = model.vocabulary().size() + 100;
        IntSequence ids = IntSequence.wrap(new int[] {bogusId});
        assertThrows(NoSuchElementException.class, () -> model.countBytes(ids));
    }

    @Test
    void countBytesNullTokensThrows() {
        TiktokenModel model = model256();
        assertThrows(NullPointerException.class, () -> model.countBytes((IntSequence) null));
    }

    // ------------------------------------------------------------------
    // IgnoreMerges: whole-token lookup bypass
    // ------------------------------------------------------------------

    @Test
    void ignoreMergesEncodesWholeTokenWhenPresent() {
        // Build a model with ignoreMerges=true and an extra whole-word token "hello"
        byte[] helloBytes = "hello".getBytes(StandardCharsets.UTF_8);
        String helloToken = ByteLevel.encode(helloBytes);
        List<String> extraTokens = new ArrayList<>();
        extraTokens.add(helloToken);
        TiktokenModel model = model256WithIgnoreMerges(extraTokens);

        IntSequence ids = model.encode("hello");
        // With ignoreMerges=true, the exact lookup can match "hello" as a whole token
        assertEquals(1, ids.length());
        assertEquals("hello", model.decode(ids));
    }

    @Test
    void ignoreMergesEncodesLongText() {
        TiktokenModel model = model256WithIgnoreMerges(List.of());
        String text = "The quick brown fox jumps over the lazy dog";
        IntSequence ids = model.encode(text);
        String decoded = model.decode(ids);
        assertEquals(text, decoded);
    }

    @Test
    void ignoreMergesWithoutMergesRoundTrip() {
        TiktokenModel model = model256WithIgnoreMerges(List.of());
        String text = "hello world test";
        IntSequence ids = model.encode(text);
        assertEquals(text, model.decode(ids));
    }

    // ------------------------------------------------------------------
    // Large input (triggers heap-based merge path)
    // ------------------------------------------------------------------

    @Test
    void largeInputRoundTrip() {
        TiktokenModel model = model256();
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 200; i++) {
            sb.append((char) ('a' + (i % 26)));
        }
        String text = sb.toString();
        IntSequence ids = model.encode(text);
        assertEquals(text, model.decode(ids));
    }

    // ------------------------------------------------------------------
    // toString
    // ------------------------------------------------------------------

    @Test
    void toStringIncludesModelName() {
        TiktokenModel model = model256();
        assertTrue(model.toString().contains("Tiktoken"));
    }

    // ------------------------------------------------------------------
    // expectedTokensPerChar
    // ------------------------------------------------------------------

    @Test
    void expectedTokensPerCharIsPositive() {
        TiktokenModel model = model256();
        assertTrue(model.expectedTokensPerChar() > 0f);
    }

    @Test
    void vocabularyReturnsNonNull() {
        TiktokenModel model = model256();
        assertTrue(model.vocabulary().size() >= 256);
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    /** Creates a minimal TiktokenModel with 256 byte-level tokens and no merges. */
    private static TiktokenModel model256() {
        String[] tokens = byteLevelTokens(256);
        Vocabulary vocabulary = vocabulary(tokens);
        return TiktokenModel.fromVocabularyAndMerges(
                vocabulary, new LongLongMap(new long[0], new long[0]));
    }

    /** Creates a model with 256+ byte-level tokens and ignoreMerges=true. */
    private static TiktokenModel model256WithIgnoreMerges(List<String> extraTokens) {
        List<String> allTokens = new ArrayList<>();
        for (int i = 0; i < 256; i++) {
            allTokens.add(String.valueOf(ByteLevel.encodeSingle((byte) i)));
        }
        for (String token : extraTokens) {
            String byteLevelToken = ByteLevel.encode(token.getBytes(StandardCharsets.UTF_8));
            allTokens.add(byteLevelToken);
        }
        Vocabulary vocabulary = vocabulary(allTokens.toArray(new String[0]));
        return TiktokenModel.fromVocabularyAndMerges(
                vocabulary, new LongLongMap(new long[0], new long[0]), true);
    }

    private static String[] byteLevelTokens(int count) {
        String[] tokens = new String[count];
        for (int i = 0; i < count; i++) {
            tokens[i] = String.valueOf(ByteLevel.encodeSingle((byte) i));
        }
        return tokens;
    }

    private static Vocabulary vocabulary(String[] tokens) {
        return new VocabularyImpl(tokens, normalTypes(tokens.length));
    }

    private static int[] normalTypes(int length) {
        int[] types = new int[length];
        for (int i = 0; i < length; i++) {
            types[i] = StandardTokenType.NORMAL.getId();
        }
        return types;
    }
}
