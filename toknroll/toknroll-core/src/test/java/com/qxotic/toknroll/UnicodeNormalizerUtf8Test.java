package com.qxotic.toknroll;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.text.Normalizer.Form;
import java.util.Arrays;
import java.util.Random;
import org.junit.jupiter.api.Test;

/** {@link UnicodeNormalizer#normalizeUtf8} gives what decoding, normalizing and encoding gives. */
class UnicodeNormalizerUtf8Test {

    /** Text that composes, reorders, decomposes, maps or stays; and bytes that are not UTF-8. */
    private static final byte[][] PIECES = {
        utf8("e"),
        utf8("<"),
        utf8(" "),
        utf8("\u0301"), // e + acute composes
        utf8("\u0338"), // < + long solidus composes
        utf8("\u0302"),
        utf8("\u0323"), // dot below sorts before circumflex
        utf8("\u00E9"),
        utf8("\u212B"), // angstrom sign, U+00C5 in NFC
        utf8("\u1100"),
        utf8("\u1161"),
        utf8("\u11A8"), // Hangul jamo, which compose into a syllable
        utf8("\uAC00"),
        utf8("\uFFFD"), // spelled, not ill-formed
        utf8("\u00A0"), // a space in NFKC
        utf8("\uFB01"), // "fi" in NFKC
        utf8("\u0344"), // decomposes in every form
        utf8("\u4E2D\u6587"),
        utf8("\uD83D\uDE00"),
        {(byte) 0xC3}, // a lead with no continuation
        {(byte) 0x80}, // a lone continuation
        {(byte) 0xED, (byte) 0xA0, (byte) 0x80}, // a surrogate
        {(byte) 0xF4, (byte) 0x90, (byte) 0x80, (byte) 0x80}, // past U+10FFFF
        {(byte) 0xC0, (byte) 0xAF}, // overlong
        {(byte) 0xF0, (byte) 0x9F, (byte) 0x98}, // truncated
    };

    @Test
    void randomTexts() {
        Random random = new Random(0x4E46L);
        for (Form form : Form.values()) {
            UnicodeNormalizer normalizer = new UnicodeNormalizer(form);
            for (int round = 0; round < 20_000; round++) {
                ByteArrayOutputStream text = new ByteArrayOutputStream();
                for (int n = random.nextInt(12); n > 0; n--) {
                    byte[] piece = PIECES[random.nextInt(PIECES.length)];
                    text.write(piece, 0, piece.length);
                }
                check(normalizer, form, text.toByteArray(), random);
            }
        }
    }

    @Test
    void sameBytesGiveNull() {
        UnicodeNormalizer nfc = new UnicodeNormalizer(Form.NFC);
        byte[] text = utf8("caf\u00E9 \uFFFD");
        assertNull(nfc.normalizeUtf8(text, 0, text.length));
        text = utf8("cafe\u0301");
        assertArrayEquals(utf8("caf\u00E9"), nfc.normalizeUtf8(text, 0, text.length));
        text = new byte[] {'a', (byte) 0xC3, 'b'};
        assertArrayEquals(utf8("a\uFFFDb"), nfc.normalizeUtf8(text, 0, text.length));
    }

    /** {@code text} as a range in a larger array, with stray bytes on both sides. */
    private static void check(UnicodeNormalizer normalizer, Form form, byte[] text, Random random) {
        int from = random.nextInt(3);
        byte[] padded = new byte[from + text.length + random.nextInt(3)];
        for (int i = 0; i < padded.length; i++) {
            padded[i] = (byte) (0x80 | random.nextInt(0x40));
        }
        System.arraycopy(text, 0, padded, from, text.length);
        byte[] expected =
                java.text.Normalizer.normalize(new String(text, StandardCharsets.UTF_8), form)
                        .getBytes(StandardCharsets.UTF_8);
        byte[] actual = normalizer.normalizeUtf8(padded, from, from + text.length);
        if (actual == null) {
            assertArrayEquals(expected, text, () -> Arrays.toString(text));
        } else {
            assertArrayEquals(expected, actual, () -> Arrays.toString(text));
            assertFalse(Arrays.equals(expected, text), () -> Arrays.toString(text));
        }
    }

    private static byte[] utf8(String s) {
        return s.getBytes(StandardCharsets.UTF_8);
    }
}
