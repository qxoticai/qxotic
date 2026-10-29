package com.qxotic.toknroll.libtoknroll;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.nio.ByteBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.util.Random;
import org.junit.jupiter.api.Test;

/** The check against the JDK's own decoder, which is what decides a U+FFFD. */
class WellFormedTest {

    private static boolean jdk(byte[] b, int from, int to) {
        try {
            StandardCharsets.UTF_8
                    .newDecoder()
                    .onMalformedInput(CodingErrorAction.REPORT)
                    .onUnmappableCharacter(CodingErrorAction.REPORT)
                    .decode(ByteBuffer.wrap(b, from, to - from));
            return true;
        } catch (CharacterCodingException e) {
            return false;
        }
    }

    private static void agree(byte[] b, int from, int to) {
        assertEquals(jdk(b, from, to), WellFormed.utf8(b, from, to), () -> hex(b, from, to));
    }

    /** The values where a rule of table 3-7 begins or ends: every check turns on one of them. */
    private static final int[] EDGES = {
        0x00, 0x7F, 0x80, 0x8F, 0x90, 0x9F, 0xA0, 0xBF, 0xC0, 0xC1, 0xC2, 0xDF, 0xE0, 0xED, 0xEF,
        0xF0, 0xF4, 0xF5, 0xFF
    };

    @Test
    void everyPairAndEveryLeadWithEdgeContinuations() {
        byte[] b = new byte[3];
        for (int x = 0; x < 256; x++) {
            b[0] = (byte) x;
            agree(b, 0, 1);
            for (int y = 0; y < 256; y++) {
                b[1] = (byte) y;
                agree(b, 0, 2);
            }
            for (int y : EDGES)
                for (int z : EDGES) {
                    b[1] = (byte) y;
                    b[2] = (byte) z;
                    agree(b, 0, 3);
                }
        }
    }

    @Test
    void everyFourByteLeadWithEdgeContinuations() {
        byte[] b = new byte[4];
        for (int x = 0xE0; x < 256; x++)
            for (int y : EDGES)
                for (int z : EDGES)
                    for (int w : EDGES) {
                        b[0] = (byte) x;
                        b[1] = (byte) y;
                        b[2] = (byte) z;
                        b[3] = (byte) w;
                        agree(b, 0, 4);
                    }
    }

    /** Mostly ASCII with a few bytes of anything, so the eight-at-a-time path meets every case. */
    @Test
    void randomTextAndRanges() {
        Random random = new Random(42);
        for (int t = 0; t < 200_000; t++) {
            byte[] b = new byte[1 + random.nextInt(40)];
            for (int i = 0; i < b.length; i++)
                b[i] =
                        (byte)
                                (random.nextInt(8) == 0
                                        ? random.nextInt(256)
                                        : 0x20 + random.nextInt(0x5F));
            if (random.nextBoolean()) {
                String s = new String(Character.toChars(random.nextInt(0x110000)));
                byte[] c = s.getBytes(StandardCharsets.UTF_8);
                System.arraycopy(c, 0, b, random.nextInt(b.length), Math.min(c.length, 1));
            }
            int from = random.nextInt(b.length);
            agree(b, from, from + random.nextInt(b.length - from + 1));
        }
    }

    @Test
    void wellFormedTextOfEveryPlane() {
        StringBuilder s = new StringBuilder();
        for (int cp = 0; cp < 0x110000; cp += 97)
            if (!Character.isSurrogate((char) cp)) s.appendCodePoint(cp);
        byte[] b = s.toString().getBytes(StandardCharsets.UTF_8);
        assertEquals(true, WellFormed.utf8(b, 0, b.length));
        agree(b, 0, b.length);
    }

    private static String hex(byte[] b, int from, int to) {
        StringBuilder s = new StringBuilder();
        for (int i = from; i < to; i++) s.append(String.format("%02X ", b[i] & 0xFF));
        return s.toString();
    }
}
