package com.qxotic.toknroll.vector;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import com.qxotic.toknroll.spi.IdBuffer;
import com.qxotic.toknroll.spi.TiktokenEngine;
import com.qxotic.toknroll.spi.TiktokenTables;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.junit.jupiter.api.Test;

class FastEncoderTest {

    /** The 256 bytes and nothing merged: a word encodes as its bytes. */
    private static final TiktokenTables BYTES =
            new TiktokenTables() {
                @Override
                public int tokenCount() {
                    return 256;
                }

                @Override
                public byte[] tokenBytes(int id) {
                    return new byte[] {(byte) id};
                }

                @Override
                public int mergeCount() {
                    return 0;
                }

                @Override
                public int mergeLeft(int merge) {
                    throw new IndexOutOfBoundsException();
                }

                @Override
                public int mergeRight(int merge) {
                    throw new IndexOutOfBoundsException();
                }

                @Override
                public int mergeResult(int merge) {
                    throw new IndexOutOfBoundsException();
                }

                @Override
                public int mergeRank(int merge) {
                    throw new IndexOutOfBoundsException();
                }

                @Override
                public boolean ignoreMerges() {
                    return false;
                }
            };

    /** The bytes and one merge, "aa" as 256. */
    private static final TiktokenTables AA =
            new TiktokenTables() {
                @Override
                public int tokenCount() {
                    return 257;
                }

                @Override
                public byte[] tokenBytes(int id) {
                    return id == 256 ? new byte[] {'a', 'a'} : new byte[] {(byte) id};
                }

                @Override
                public int mergeCount() {
                    return 1;
                }

                @Override
                public int mergeLeft(int merge) {
                    return 'a';
                }

                @Override
                public int mergeRight(int merge) {
                    return 'a';
                }

                @Override
                public int mergeResult(int merge) {
                    return 256;
                }

                @Override
                public int mergeRank(int merge) {
                    return 0;
                }

                @Override
                public boolean ignoreMerges() {
                    return false;
                }
            };

    /** A scratch lives on across calls, so its edge stamps may have counted up to anything. */
    @Test
    void aLargeMergeIsTheSameWhateverStampTheScratchReached() {
        FastEncoder encoder = new FastEncoder(AA, BitSplit.forRules(TiktokenEngine.CL100K));
        byte[] word = "a".repeat(200).getBytes(StandardCharsets.UTF_8);
        int[] expected = new int[100];
        Arrays.fill(expected, 256);
        for (int stamp : new int[] {1, -50, -1, Integer.MAX_VALUE - 50}) {
            FastEncoder.Scratch s = new FastEncoder.Scratch(null);
            s.stamp = stamp;
            IdBuffer out = new IdBuffer(16);
            encoder.merge(word, 0, word.length, out, s);
            assertArrayEquals(expected, Arrays.copyOf(out.array(), out.size()), "from " + stamp);
        }
    }

    @Test
    void manyThreadsAtOnceGetTheSameIds() throws Exception {
        FastEncoder encoder = new FastEncoder(BYTES, BitSplit.forRules(TiktokenEngine.CL100K));
        String text = "the cat sat on the mat, and the dog sat on the log. ".repeat(2000);
        int[] expected = ids(text);
        try (var threads = Executors.newVirtualThreadPerTaskExecutor()) {
            List<Future<int[]>> calls = new ArrayList<>();
            for (int i = 0; i < 64 * Runtime.getRuntime().availableProcessors(); i++) {
                calls.add(
                        threads.submit(
                                () -> {
                                    IdBuffer out = new IdBuffer(16);
                                    encoder.encode(text, 0, text.length(), out);
                                    return Arrays.copyOf(out.array(), out.size());
                                }));
            }
            for (Future<int[]> call : calls) {
                assertArrayEquals(expected, call.get());
            }
        }
    }

    @Test
    void aGiantWordLeavesTheEncoderWorking() {
        FastEncoder encoder = new FastEncoder(BYTES, BitSplit.forRules(TiktokenEngine.CL100K));
        String giant = "x".repeat(1 << 20);
        for (String text : new String[] {"a few words", giant, "a few words"}) {
            IdBuffer out = new IdBuffer(16);
            encoder.encode(text, 0, text.length(), out);
            assertArrayEquals(ids(text), Arrays.copyOf(out.array(), out.size()));
        }
    }

    private static int[] ids(String text) {
        byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
        int[] ids = new int[bytes.length];
        for (int i = 0; i < bytes.length; i++) {
            ids[i] = bytes[i] & 0xFF;
        }
        return ids;
    }
}
