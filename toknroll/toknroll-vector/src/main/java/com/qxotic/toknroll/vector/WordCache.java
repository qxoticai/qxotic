package com.qxotic.toknroll.vector;

import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.nio.ByteOrder;

/**
 * The token ids of the words one thread has encoded, so a repeat costs a probe instead of a merge.
 *
 * <p>A word of up to {@link #MAX_BYTES} bytes is its own key, packed with its length into one long
 * when it is short and two when it is not, so a hit is exact by construction; longer words, and
 * words of more than {@link #MAX_IDS} ids, are not cached. A hash of the key picks a home slot and
 * a one-byte tag. The tags live in a row of their own: a lookup loads the eight tags from the home
 * slot in one go and tests them against the tag at once, then confirms a candidate against its key.
 * A miss inserts into the first empty slot of that window, or the home slot when the window is
 * full.
 *
 * <p>The table starts small and doubles, up to the slots it is given, each time it has taken as
 * many words as it has slots: a cache that has seen a few documents takes a few megabytes, one that
 * works through a corpus grows to its cap, where a Zipf vocabulary's tail still hits. A slot takes
 * 32 bytes and its tag one more. A doubling rehashes what the table holds; nothing else here
 * allocates.
 *
 * <p>After Swiss tables (the tag row) and the cache of tokenizers 1.0 (the packed keys, the ids
 * inline).
 */
final class WordCache {
    static final int MAX_BYTES = 15;
    static final int MAX_IDS = 4;

    private static final int FIRST_SLOTS = 1 << 16;
    private static final int WINDOW = 8;
    private static final VarHandle LONGS =
            MethodHandles.byteArrayViewVarHandle(long[].class, ByteOrder.LITTLE_ENDIAN);
    private static final long LOW_BYTES = 0x0101010101010101L;
    private static final long HIGH_BITS = 0x8080808080808080L;
    private static final long COUNT_MASK = 7L << 60; // the id count rides in the key's spare bits

    private final int maxSlots;
    private int mask;
    private byte[] tags; // 0 is empty

    /**
     * Four longs per slot: the key, then the four ids packed in two. A long array's data starts 16
     * bytes into its cache line, so the two leading longs put every 32-byte slot inside one line.
     */
    private long[] slots;

    private int inserted; // since the last doubling

    /** A cache that grows up to {@code maxSlots}, a power of two. */
    WordCache(int maxSlots) {
        this.maxSlots = maxSlots;
        int first = Math.min(FIRST_SLOTS, maxSlots);
        this.mask = first - 1;
        this.tags = new byte[first + WINDOW];
        this.slots = new long[4 * (first + WINDOW) + 2];
    }

    private int victim;

    // The last lookup's word: its key, its tag, and the slot an insert of it takes.
    private long lo;
    private long hi;
    private byte tag;
    private int slot;

    /**
     * The ids of {@code text[start, end)} written to {@code ids[at, at + count)}, which must have
     * room for {@link #MAX_IDS}, and their count; 0 when the cache does not hold the word, and then
     * {@link #insert} may follow for it. The text must have {@link FastEncoder#PADDING} readable
     * bytes past {@code end}.
     */
    int lookup(byte[] text, int start, int end, int[] ids, int at) {
        int length = end - start;
        long lo;
        long hi;
        if (length < 8) {
            // one long: the bytes, the length in the top byte
            lo = ((long) LONGS.get(text, start) & mask(length)) | ((long) length << 56);
            hi = 0;
        } else {
            lo = (long) LONGS.get(text, start);
            hi = ((long) LONGS.get(text, start + 8) & mask(length - 8)) | ((long) length << 56);
        }
        long hash = hash(lo, hi);
        int home = (int) hash & mask;
        long tag = (hash >>> 56) | 1;
        long row = (long) LONGS.get(tags, home);
        long same = zeroBytes(row ^ (tag * LOW_BYTES));
        long[] s = slots;
        while (same != 0) {
            int slot = 4 * (home + (Long.numberOfTrailingZeros(same) >>> 3)) + 2;
            long counted = s[slot + 1];
            if (s[slot] == lo && (counted & ~COUNT_MASK) == hi) {
                long first = s[slot + 2];
                long second = s[slot + 3];
                // four stores, no branch: the count says how many are meant
                ids[at] = (int) first;
                ids[at + 1] = (int) (first >>> 32);
                ids[at + 2] = (int) second;
                ids[at + 3] = (int) (second >>> 32);
                return (int) (counted >>> 60);
            }
            same &= same - 1;
        }
        this.lo = lo;
        this.hi = hi;
        this.tag = (byte) tag;
        this.slot = free(home, row);
        return 0;
    }

    /**
     * Encodes the pieces of a batch from {@code k} on, {@code ends[k]} the end of each and {@code
     * cursor} the start of the first, as far as this cache and the {@code singleByte} table answer
     * them, writing the ids to {@code ids} from {@code size} on (room for {@link #MAX_IDS} per
     * piece). Returns the index of the first piece it could not answer, or {@code n}, in the high
     * half and the size reached in the low half.
     *
     * <p>This loop is a method of its own so that the JIT compiles it with the merge out of sight:
     * inlined next to the miss path it spilled a third of its values to the stack.
     */
    long run(
            byte[] text,
            int[] ends,
            int k,
            int n,
            int cursor,
            int[] ids,
            int size,
            int[] singleByte) {
        for (; k < n; k++) {
            int e = ends[k];
            int length = e - cursor;
            if (length == 1) {
                ids[size++] = singleByte[text[cursor] & 0xFF];
            } else {
                int hit = length <= MAX_BYTES ? lookup(text, cursor, e, ids, size) : 0;
                if (hit == 0) {
                    break;
                }
                size += hit;
            }
            cursor = e;
        }
        return ((long) k << 32) | size;
    }

    /**
     * The slot an insert takes in the window at {@code home}, whose tags are {@code row}: the first
     * empty one, else a rotating one. A full window evicts a rotating slot, not always the home
     * one, or two frequent words with one home would take turns evicting each other.
     */
    private int free(int home, long row) {
        long empty = zeroBytes(row);
        return empty != 0
                ? home + (Long.numberOfTrailingZeros(empty) >>> 3)
                : home + (victim++ & (WINDOW - 1));
    }

    /**
     * Records the word of the last {@link #lookup} as the {@code count} ids at {@code ids[first]}.
     */
    void insert(int[] ids, int first, int count) {
        if (++inserted > mask && mask < maxSlots - 1) {
            grow();
        }
        int slot = this.slot;
        tags[slot] = tag;
        int at = 4 * slot + 2;
        slots[at] = lo;
        slots[at + 1] = hi | ((long) count << 60);
        slots[at + 2] = (ids[first] & 0xFFFFFFFFL) | (count > 1 ? (long) ids[first + 1] << 32 : 0);
        slots[at + 3] =
                count > 2
                        ? (ids[first + 2] & 0xFFFFFFFFL)
                                | (count > 3 ? (long) ids[first + 3] << 32 : 0)
                        : 0;
    }

    /** Twice the slots, the words rehashed into them; the pending insert keeps its place. */
    private void grow() {
        byte[] oldTags = tags;
        long[] oldSlots = slots;
        int oldSlotCount = mask + 1;
        int slotCount = oldSlotCount << 1;
        mask = slotCount - 1;
        tags = new byte[slotCount + WINDOW];
        slots = new long[4 * (slotCount + WINDOW) + 2];
        inserted = 0;
        for (int from = 0; from < oldSlotCount + WINDOW; from++) {
            if (oldTags[from] == 0) {
                continue;
            }
            int at = 4 * from + 2;
            long lo = oldSlots[at];
            long hi = oldSlots[at + 1];
            int to = place(lo, hi & ~COUNT_MASK);
            tags[to] = oldTags[from];
            System.arraycopy(oldSlots, at, slots, 4 * to + 2, 4);
            inserted++;
        }
        // the word being inserted was placed against the old table
        this.slot = place(lo, hi);
    }

    /** The slot a word with this key takes in the current table. */
    private int place(long lo, long hi) {
        int home = (int) hash(lo, hi) & mask;
        return free(home, (long) LONGS.get(tags, home));
    }

    /** For a short word hi is zero and this is one multiply, which the JIT sees. */
    private static long hash(long lo, long hi) {
        long hash = (lo * 0x9E3779B97F4A7C15L) ^ (hi * 0xC2B2AE3D27D4EB4FL);
        return hash ^ (hash >>> 29);
    }

    /**
     * The bytes of a long that are zero, as their high bits. Only the lowest set bit is exact: a
     * byte above a true zero may be flagged too, which costs a key compare, never a wrong answer.
     */
    private static long zeroBytes(long x) {
        return (x - LOW_BYTES) & ~x & HIGH_BITS;
    }

    /** The low {@code n} bytes of a long, for {@code n} in 1..8. */
    private static long mask(int n) {
        return n >= 8 ? -1L : (1L << (8 * n)) - 1;
    }
}
