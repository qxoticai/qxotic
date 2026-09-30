package com.qxotic.toknroll.impl;

import com.qxotic.toknroll.IntSequence;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.nio.ByteOrder;

/**
 * The token ids of the words one thread has encoded, so a repeat costs a probe instead of a merge.
 *
 * <p>A word of up to {@link #MAX_BYTES} bytes is its own key, packed into two longs with its
 * length, so a hit is exact by construction; longer words, and words of more than {@link #MAX_IDS}
 * ids, are not cached. A hash of the key picks a home slot and a one-byte tag. The tags live in a
 * row of their own: a lookup loads the eight tags from the home slot in one go and tests them
 * against the tag at once, then confirms a candidate against its key. A miss inserts into the first
 * empty slot of that window, or the home slot when the window is full. Nothing here allocates after
 * construction.
 *
 * <p>After Swiss tables (the tag row) and the cache of tokenizers 1.0 (the packed keys, the ids
 * inline).
 */
final class WordCache {
    static final int MAX_BYTES = 15;
    static final int MAX_IDS = 4;

    private static final int SLOTS = 1 << 17;
    private static final int WINDOW = 8;
    private static final VarHandle LONGS =
            MethodHandles.byteArrayViewVarHandle(long[].class, ByteOrder.LITTLE_ENDIAN);
    private static final long LOW_BYTES = 0x0101010101010101L;
    private static final long HIGH_BITS = 0x8080808080808080L;
    private static final long COUNT_MASK = 7L << 60; // the id count rides in the key's spare bits

    private final byte[] tags = new byte[SLOTS + WINDOW]; // 0 is empty

    /**
     * Four longs per slot: the key, then the four ids packed in two. A long array's data starts 16
     * bytes into its cache line, so the two leading longs put every 32-byte slot inside one line.
     */
    private final long[] slots = new long[4 * (SLOTS + WINDOW) + 2];

    private long lo;
    private long hi;
    private byte tag;
    private int victim;
    private int slot; // where the last lookup found its word, or where it should go

    /**
     * The ids of {@code text[start, end)} appended to {@code out} and their count, or 0 when the
     * cache does not hold the word. Then {@link #insert} may follow, for the same word. The text
     * must have {@link TiktokenSplitter#PADDING} readable bytes past {@code end}.
     */
    int lookup(byte[] text, int start, int end, IntSequence.Builder out) {
        int at = find(text, start, end);
        if (at < 0) {
            return 0;
        }
        int count = (int) (slots[at + 1] >>> 60);
        long first = slots[at + 2];
        out.add((int) first);
        if (count > 1) {
            out.add((int) (first >>> 32));
            if (count > 2) {
                long second = slots[at + 3];
                out.add((int) second);
                if (count > 3) {
                    out.add((int) (second >>> 32));
                }
            }
        }
        return count;
    }

    /**
     * {@link #lookup} writing the ids to {@code ids[at, at + count)}, which must have room for
     * {@link #MAX_IDS}.
     */
    int lookup(byte[] text, int start, int end, int[] ids, int at) {
        int slot = find(text, start, end);
        if (slot < 0) {
            return 0;
        }
        int count = (int) (slots[slot + 1] >>> 60);
        long first = slots[slot + 2];
        long second = slots[slot + 3];
        // four stores, no branch: the count says how many are meant
        ids[at] = (int) first;
        ids[at + 1] = (int) (first >>> 32);
        ids[at + 2] = (int) second;
        ids[at + 3] = (int) (second >>> 32);
        return count;
    }

    /** The index in {@link #slots} of the word's slot, or -1 with the slot an insert would take. */
    private int find(byte[] text, int start, int end) {
        int length = end - start;
        long lo = (long) LONGS.get(text, start) & mask(length);
        long hi = length > 8 ? (long) LONGS.get(text, start + 8) & mask(length - 8) : 0;
        hi |= (long) length << 56;
        this.lo = lo;
        this.hi = hi;
        long hash = (lo * 0x9E3779B97F4A7C15L) ^ (hi * 0xC2B2AE3D27D4EB4FL);
        hash ^= hash >>> 29;
        int home = (int) hash & (SLOTS - 1);
        long tag = (hash >>> 56) | 1;
        long row = (long) LONGS.get(tags, home);
        long same = zeroBytes(row ^ (tag * LOW_BYTES));
        while (same != 0) {
            int slot = home + (Long.numberOfTrailingZeros(same) >>> 3);
            int at = 4 * slot + 2;
            if (slots[at] == lo && (slots[at + 1] & ~COUNT_MASK) == hi) {
                return at;
            }
            same &= same - 1;
        }
        long empty = zeroBytes(row);
        // A full window evicts a rotating slot, not always the home one: two frequent words with
        // one home would otherwise take turns evicting each other.
        this.slot =
                empty != 0
                        ? home + (Long.numberOfTrailingZeros(empty) >>> 3)
                        : home + (victim++ & (WINDOW - 1));
        this.tag = (byte) tag;
        return -1;
    }

    /** Records the word of the last {@link #lookup} as {@code count} ids taken from {@code out}. */
    void insert(IntSequence.Builder out, int first, int count) {
        int slot = this.slot;
        tags[slot] = tag;
        int at = 4 * slot + 2;
        slots[at] = lo;
        slots[at + 1] = hi | ((long) count << 60);
        slots[at + 2] =
                (id(out, first) & 0xFFFFFFFFL) | (count > 1 ? (long) id(out, first + 1) << 32 : 0);
        slots[at + 3] =
                count > 2
                        ? (id(out, first + 2) & 0xFFFFFFFFL)
                                | (count > 3 ? (long) id(out, first + 3) << 32 : 0)
                        : 0;
    }

    /** An id already appended to {@code out}, without a view object for the model's own builder. */
    private static int id(IntSequence.Builder out, int index) {
        return out instanceof IntSequenceBuilder
                ? ((IntSequenceBuilder) out).get(index)
                : out.asSequenceView().intAt(index);
    }

    /** The bytes of a long that are zero, as their high bits. */
    private static long zeroBytes(long x) {
        return (x - LOW_BYTES) & ~x & HIGH_BITS;
    }

    /** The low {@code n} bytes of a long, for {@code n} in 1..8. */
    private static long mask(int n) {
        return n >= 8 ? -1L : (1L << (8 * n)) - 1;
    }
}
