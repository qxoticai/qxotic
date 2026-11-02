package com.qxotic.toknroll.impl;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Normalizer;
import com.qxotic.toknroll.StandardTokenType;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Toknroll;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/**
 * The SentencePiece wrapper: Metaspace's prepend on the way in, each token's bytes on the way out.
 */
class MetaspaceTokenizerTest {

    private static final String[] WORDS = {
        "▁", "h", "e", "l", "o", "w", "r", "d", "he", "ll", "llo", "hello", "▁hello", "or", "ld",
        "orld", "world", "▁world"
    };

    /** 256 byte tokens, then {@link #WORDS}, merged by score. */
    private static final SentencePieceBpeModel MODEL = model();

    /** The pipeline the loaders build: a normalizer spells the spaces U+2581. */
    private static final Tokenizer BASE =
            Toknroll.pipeline((Normalizer) text -> text.toString().replace(' ', '▁'), MODEL);

    private static SentencePieceBpeModel model() {
        int n = 256 + WORDS.length;
        String[] tokens = new String[n];
        int[] types = new int[n];
        float[] scores = new float[n];
        for (int b = 0; b < 256; b++) {
            tokens[b] = String.format("<0x%02X>", b);
            types[b] = StandardTokenType.BYTE.getId();
        }
        for (int i = 0; i < WORDS.length; i++) {
            tokens[256 + i] = WORDS[i];
            types[256 + i] = StandardTokenType.NORMAL.getId();
            scores[256 + i] = WORDS[i].length();
        }
        return SentencePieceBpeModel.fromVocabulary(new VocabularyImpl(tokens, types), scores);
    }

    private static Tokenizer metaspace(String scheme, boolean trim) {
        return new MetaspaceTokenizer(BASE, scheme, trim);
    }

    private static int[] ids(Tokenizer tokenizer, String text) {
        return tokenizer.encode(text).toArray();
    }

    private static int[] model(String text) {
        return MODEL.encode(text).toArray();
    }

    // ---- encode: Metaspace's prepend_scheme ----

    @Test
    void alwaysAndFirstPrependBeforeTheText() {
        for (String scheme : new String[] {"always", "first"}) {
            assertArrayEquals(model("▁hello▁world"), ids(metaspace(scheme, true), "hello world"));
        }
        assertArrayEquals(model("hello▁world"), ids(metaspace("never", true), "hello world"));
    }

    @Test
    void textStartingWithASpaceGetsNoSecondOne() {
        // Hugging Face: " hello" is "▁hello", not "▁▁hello"
        for (String scheme : new String[] {"always", "first", "never"}) {
            Tokenizer tokenizer = metaspace(scheme, true);
            assertArrayEquals(model("▁hello"), ids(tokenizer, " hello"), scheme);
            assertArrayEquals(model("▁hello"), ids(tokenizer, "▁hello"), scheme);
            assertArrayEquals(new int[0], ids(tokenizer, ""), scheme);
        }
    }

    @Test
    void firstPrependsAtTheStartOfTheTextOnly() {
        String text = "hello world";
        IntSequence.Builder first = IntSequence.newBuilder();
        metaspace("first", true).encodeInto(text, 6, text.length(), first);
        assertArrayEquals(model("world"), first.build().toArray());
        IntSequence.Builder always = IntSequence.newBuilder();
        metaspace("always", true).encodeInto(text, 6, text.length(), always);
        assertArrayEquals(model("▁world"), always.build().toArray());
    }

    @ParameterizedTest
    @ValueSource(strings = {"always", "first", "never"})
    void countTokensIsTheEncodedLength(String scheme) {
        Tokenizer tokenizer = metaspace(scheme, true);
        for (String text : new String[] {"", " ", "hello world", " hello", "x😀 y", "▁▁a"}) {
            assertEquals(ids(tokenizer, text).length, tokenizer.countTokens(text), text);
        }
    }

    @Test
    void anUnknownSchemeIsRefused() {
        assertThrows(IllegalArgumentException.class, () -> metaspace("sometimes", true));
    }

    // ---- decode: each token's bytes ----

    @Test
    void trimDropsTheSpaceTheFirstTokenStartsWith() {
        IntSequence ids = metaspace("always", true).encode("hello world");
        assertEquals("hello world", metaspace("always", true).decode(ids));
        assertEquals(" hello world", metaspace("always", false).decode(ids));
        // only the first token's: a later one keeps its space
        IntSequence later = IntSequence.wrap(new int[] {model("h")[0], model("▁world")[0]});
        assertEquals("h world", metaspace("never", true).decode(later));
    }

    @Test
    void aLoneByteTokenIsItsByteNotAReplacementChar() {
        IntSequence half = IntSequence.wrap(new int[] {0xC3}); // the first byte of "é"
        Tokenizer tokenizer = metaspace("never", true);
        assertArrayEquals(new byte[] {(byte) 0xC3}, tokenizer.decodeBytes(half));
        assertEquals(1, tokenizer.countBytes(half));
    }

    @Test
    void aBufferOfTheCountedSizeHoldsTheBytes() {
        Tokenizer tokenizer = metaspace("never", true);
        IntSequence ids = tokenizer.encode("é"); // two byte tokens, C3 A9
        assertEquals(2, tokenizer.countBytes(ids));
        ByteBuffer exact = ByteBuffer.allocate(2);
        assertEquals(2, tokenizer.decodeBytesInto(ids, 0, exact));
        assertArrayEquals(new byte[] {(byte) 0xC3, (byte) 0xA9}, exact.array());
    }

    @Test
    void decodingOneByteAtATimeKeepsEveryByte() {
        for (boolean trim : new boolean[] {true, false}) {
            Tokenizer tokenizer = metaspace("always", trim);
            IntSequence ids = tokenizer.encode("a é 😀 日本 hello");
            byte[] expected = tokenizer.decodeBytes(ids);
            ByteBuffer all = ByteBuffer.allocate(expected.length);
            for (int i = 0; i < ids.length(); ) {
                ByteBuffer room = ByteBuffer.allocate(8); // wider than any token here
                i += tokenizer.decodeBytesInto(ids, i, room);
                all.put(room.flip());
            }
            assertArrayEquals(expected, all.array());
        }
    }

    /** The two decodes agree on any ids, byte tokens and partial characters included. */
    @Test
    void decodeIsTheUtf8OfDecodeBytes() {
        Random random = new Random(7);
        for (boolean trim : new boolean[] {true, false}) {
            Tokenizer tokenizer = metaspace("first", trim);
            for (int n = 0; n < 2_000; n++) {
                List<Integer> list = new ArrayList<>();
                for (int k = random.nextInt(8); k > 0; k--) {
                    list.add(random.nextInt(256 + WORDS.length));
                }
                IntSequence ids =
                        IntSequence.wrap(list.stream().mapToInt(Integer::intValue).toArray());
                byte[] bytes = tokenizer.decodeBytes(ids);
                assertEquals(new String(bytes, StandardCharsets.UTF_8), tokenizer.decode(ids));
                assertEquals(bytes.length, tokenizer.countBytes(ids));
            }
        }
    }

    @Test
    void anUnknownIdIsRefused() {
        IntSequence bogus = IntSequence.wrap(new int[] {256 + WORDS.length + 7});
        Tokenizer tokenizer = metaspace("never", true);
        assertThrows(IllegalArgumentException.class, () -> tokenizer.decodeBytes(bogus));
        assertThrows(IllegalArgumentException.class, () -> tokenizer.countBytes(bogus));
    }
}
