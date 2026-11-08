package com.qxotic.toknroll.vector;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;

import com.qxotic.toknroll.spi.TiktokenTables;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

/** The proof behind the seeds, on vocabularies small enough to see through. */
class SeedsTest {

    private static final byte[] ZHONG = "中".getBytes(StandardCharsets.UTF_8); // E4 B8 AD
    private static final byte[] WEN = "文".getBytes(StandardCharsets.UTF_8); // E6 96 87
    private static final int ZHONG_ID = 257;
    private static final int WEN_ID = 259;

    /** The bytes, the two characters built up over two merges each, and what {@code more} adds. */
    private static TiktokenTables model(Object[]... more) {
        List<byte[]> tokens = new ArrayList<>();
        for (int b = 0; b < 256; b++) {
            tokens.add(new byte[] {(byte) b});
        }
        List<int[]> merges = new ArrayList<>(); // left, right, result, rank
        tokens.add(Arrays.copyOf(ZHONG, 2)); // 256
        tokens.add(ZHONG); // 257
        tokens.add(Arrays.copyOf(WEN, 2)); // 258
        tokens.add(WEN); // 259
        merges.add(new int[] {0xE4, 0xB8, 256, 10});
        merges.add(new int[] {256, 0xAD, 257, 11});
        merges.add(new int[] {0xE6, 0x96, 258, 12});
        merges.add(new int[] {258, 0x87, 259, 13});
        for (Object[] m : more) {
            // a token from its bytes, merged from the two ids, at the rank
            tokens.add((byte[]) m[0]);
            merges.add(new int[] {(int) m[1], (int) m[2], tokens.size() - 1, (int) m[3]});
        }
        return new TiktokenTables() {
            @Override
            public int tokenCount() {
                return tokens.size();
            }

            @Override
            public byte[] tokenBytes(int id) {
                return tokens.get(id);
            }

            @Override
            public int mergeCount() {
                return merges.size();
            }

            @Override
            public int mergeLeft(int merge) {
                return merges.get(merge)[0];
            }

            @Override
            public int mergeRight(int merge) {
                return merges.get(merge)[1];
            }

            @Override
            public int mergeResult(int merge) {
                return merges.get(merge)[2];
            }

            @Override
            public int mergeRank(int merge) {
                return merges.get(merge)[3];
            }

            @Override
            public boolean ignoreMerges() {
                return false;
            }
        };
    }

    private static Seeds prove(TiktokenTables model) {
        int[] singleByte = new int[256];
        for (int b = 0; b < 256; b++) {
            singleByte[b] = b;
        }
        byte[][] tokenBytes = new byte[model.tokenCount()][];
        for (int id = 0; id < tokenBytes.length; id++) {
            tokenBytes[id] = model.tokenBytes(id);
        }
        return Seeds.prove(model, tokenBytes, new PairMap(model), singleByte);
    }

    private static int[] fill(Seeds seeds, String text) {
        byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
        int[] tokens = new int[bytes.length];
        int n = seeds.fill(bytes, 0, bytes.length, tokens);
        return Arrays.copyOf(tokens, n);
    }

    private static byte[] bytes(int... values) {
        byte[] out = new byte[values.length];
        for (int i = 0; i < values.length; i++) {
            out[i] = (byte) values[i];
        }
        return out;
    }

    @Test
    void charactersWhoseBytesMergeIntoTheirTokenAreSeeded() {
        Seeds seeds =
                prove(
                        model(
                                new Object[] {
                                    bytes(0xE4, 0xB8, 0xAD, 0xE6, 0x96, 0x87), 257, 259, 20
                                }));
        assertNotNull(seeds);
        assertEquals(2, seeds.count());
        assertArrayEquals(new int[] {ZHONG_ID, WEN_ID}, fill(seeds, "中文"));
        assertArrayEquals(new int[] {'a', ZHONG_ID, 'b'}, fill(seeds, "a中b"));
    }

    @Test
    void aMergeOfWholeCharactersThatOutranksTheirOwnForbidsTheSeed() {
        // 中文 at rank 5: the byte loop has neither character ready when the seeded loop would merge
        Seeds seeds =
                prove(model(new Object[] {bytes(0xE4, 0xB8, 0xAD, 0xE6, 0x96, 0x87), 257, 259, 5}));
        assertNull(seeds);
    }

    @Test
    void aStraddlingMergeForbidsTheSeedBesideItsByteOnly() {
        // a space with the lead byte of 中, before the lead byte merges with the rest
        Seeds seeds = prove(model(new Object[] {bytes(0x20, 0xE4), 0x20, 0xE4, 5}));
        assertNotNull(seeds);
        assertEquals(2, seeds.count());
        assertArrayEquals(new int[] {0x20, 0xE4, 0xB8, 0xAD}, fill(seeds, " 中"));
        assertArrayEquals(new int[] {'a', ZHONG_ID}, fill(seeds, "a中"));
        assertArrayEquals(new int[] {ZHONG_ID}, fill(seeds, "中"));
    }

    @Test
    void aStraddlingMergeAfterTheCharacterForbidsTheSeedBeforeItsByte() {
        // the last byte of 中 with the lead byte of 文, before 中 takes its last byte
        // and the lead byte of 文 after the last byte of 中, before 文 takes its lead byte
        Seeds seeds = prove(model(new Object[] {bytes(0xAD, 0xE6), 0xAD, 0xE6, 5}));
        assertNotNull(seeds);
        assertEquals(2, seeds.count());
        assertArrayEquals(new int[] {0xE4, 0xB8, 0xAD, 0xE6, 0x96, 0x87}, fill(seeds, "中文"));
        assertArrayEquals(new int[] {ZHONG_ID, 'x'}, fill(seeds, "中x"));
        assertArrayEquals(new int[] {'x', WEN_ID}, fill(seeds, "x文"));
    }

    @Test
    void aStraddlingMergeThatTheCharacterOutranksChangesNothing() {
        // the same straddling merges at rank 30, after both characters are whole
        Seeds seeds =
                prove(
                        model(
                                new Object[] {bytes(0x20, 0xE4), 0x20, 0xE4, 30},
                                new Object[] {bytes(0xAD, 0xE6), 0xAD, 0xE6, 31}));
        assertNotNull(seeds);
        assertArrayEquals(new int[] {0x20, ZHONG_ID, WEN_ID}, fill(seeds, " 中文"));
    }

    @Test
    void illFormedAndSupplementaryBytesStayBytes() {
        // and U+07FF as a token, so that its overlong form has a seed to be denied
        Seeds seeds = prove(model(new Object[] {bytes(0xDF, 0xBF), 0xDF, 0xBF, 14}));
        assertEquals(3, seeds.count());
        assertArrayEquals(new int[] {ZHONG_ID}, fill(seeds, "中"));
        assertArrayEquals(new int[] {260}, fill(seeds, "߿"));
        byte[] cut = Arrays.copyOf(ZHONG, 2);
        int[] tokens = new int[2];
        assertEquals(2, seeds.fill(cut, 0, 2, tokens));
        assertArrayEquals(new int[] {0xE4, 0xB8}, tokens);
        byte[] overlong = bytes(0xE0, 0x9F, 0xBF); // U+07FF in three bytes: not its token's bytes
        tokens = new int[3];
        assertEquals(3, seeds.fill(overlong, 0, 3, tokens));
        assertArrayEquals(new int[] {0xE0, 0x9F, 0xBF}, tokens);
        byte[] emoji = "😀".getBytes(StandardCharsets.UTF_8);
        assertEquals(4, seeds.fill(emoji, 0, 4, new int[4]));
    }
}
