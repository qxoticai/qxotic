package com.qxotic.toknroll.vector;

import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.nio.ByteOrder;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * The token ids of the words a tokenizer has encoded, so a repeat costs a probe instead of a merge.
 * One cache per tokenizer, read and written by every thread that encodes with it.
 *
 * <p>A word of up to {@link #MAX_BYTES} bytes is its own key, packed with its length into one long
 * when it is short and two when it is not, so a hit is exact by construction; longer words, and
 * words of more than {@link #MAX_IDS} ids, are not cached. A hash of the key picks a home slot and
 * a one-byte tag. The tags live in a row of their own: a lookup loads the eight tags from the home
 * slot in one go and tests them against the tag at once, then confirms a candidate against its key.
 *
 * <p>Threads share the table without a lock because a slot never changes once its tag is out: a
 * miss claims an empty slot of its window by turning its tag from empty to busy, writes the key and
 * the ids, and only then publishes the tag. A lookup reads the tags before the slot, so a tag it
 * matches stands for a finished entry. A word whose window is full is not cached.
 *
 * <p>The table starts small and is replaced, never changed in place: by one twice its size with
 * what it holds when its windows fill, up to the slots it is given, and once at that size by an
 * empty one when as many words have found no room as it has slots, so the cache follows the text it
 * sees. A slot takes 32 bytes and its tag one more. A thread that still reads a replaced table
 * reads it whole, and what it inserts there is lost, which a cache can afford.
 *
 * <p>After Swiss tables (the tag row) and the cache of tokenizers 1.0 (the packed keys, the ids
 * inline).
 */
final class WordCache {
    static final int MAX_BYTES = 15;
    static final int MAX_IDS = 4;

    private static final int FIRST_SLOTS = 1 << 16;
    private static final int WINDOW = 8;
    private static final byte EMPTY = 0;
    private static final byte BUSY = 2; // tags are odd, so a busy slot matches no lookup
    private static final VarHandle LONGS =
            MethodHandles.byteArrayViewVarHandle(long[].class, ByteOrder.LITTLE_ENDIAN);
    private static final VarHandle TAG = MethodHandles.arrayElementVarHandle(byte[].class);
    private static final long LOW_BYTES = 0x0101010101010101L;
    private static final long HIGH_BITS = 0x8080808080808080L;
    private static final long COUNT_MASK = 7L << 60; // the id count rides in the key's spare bits

    private final int maxSlots;
    private volatile Table table;
    private final AtomicBoolean replacing = new AtomicBoolean();

    /** A cache that grows up to {@code maxSlots}, a power of two. */
    WordCache(int maxSlots) {
        this.maxSlots = maxSlots;
        this.table = new Table(Math.min(FIRST_SLOTS, maxSlots));
    }

    private static final class Table {
        final int mask;
        final byte[] tags; // 0 is empty

        /**
         * Four longs per slot: the key, then the four ids packed in two. A long array's data starts
         * 16 bytes into its cache line, so the two leading longs put every 32-byte slot inside one
         * line.
         */
        final long[] slots;

        int homeless; // words that found their window full: counted loosely, a race loses a few

        Table(int slots) {
            this.mask = slots - 1;
            this.tags = new byte[slots + WINDOW];
            this.slots = new long[4 * (slots + WINDOW) + 2];
        }
    }

    /** What one call keeps of the cache: the table it reads and the key of its last miss. */
    static final class Probe {
        private Table table;
        private long lo;
        private long hi;
        private byte tag;
        private int slot; // the empty slot an insert tries, or -1 when the window is full
    }

    /** A probe of the current table; {@link #refresh} it to follow a replacement. */
    Probe probe() {
        Probe p = new Probe();
        p.table = table;
        return p;
    }

    void refresh(Probe p) {
        p.table = table;
    }

    /**
     * The ids of {@code text[start, end)} written to {@code ids[at, at + count)}, which must have
     * room for {@link #MAX_IDS}, and their count; 0 when the cache does not hold the word, and then
     * {@link #insert} may follow for it with the same probe. The text must have {@link
     * FastEncoder#PADDING} readable bytes past {@code end}.
     */
    static int lookup(
            byte[] text,
            int start,
            int end,
            int[] ids,
            int at,
            int mask,
            byte[] tags,
            long[] s,
            Probe p) {
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
        VarHandle.acquireFence(); // the slots are read after the tags that published them
        long same = zeroBytes(row ^ (tag * LOW_BYTES));
        while (same != 0) {
            int j = Long.numberOfTrailingZeros(same) >>> 3;
            int slot = 4 * (home + j) + 2;
            long counted = s[slot + 1];
            // a byte above a true match may be flagged too, and may be a slot still being
            // written: only a tag that is out vouches for its slot
            if (((row >>> (j << 3)) & 0xFF) == tag
                    && s[slot] == lo
                    && (counted & ~COUNT_MASK) == hi) {
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
        long empty = zeroBytes(row);
        p.lo = lo;
        p.hi = hi;
        p.tag = (byte) tag;
        p.slot = empty != 0 ? home + (Long.numberOfTrailingZeros(empty) >>> 3) : -1;
        return 0;
    }

    /** {@link #lookup} in the probe's table. */
    static int lookup(byte[] text, int start, int end, int[] ids, int at, Probe p) {
        Table t = p.table;
        return lookup(text, start, end, ids, at, t.mask, t.tags, t.slots, p);
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
    static long run(
            byte[] text,
            int[] ends,
            int k,
            int n,
            int cursor,
            int[] ids,
            int size,
            int[] singleByte,
            Probe p) {
        Table t = p.table;
        int mask = t.mask;
        byte[] tags = t.tags;
        long[] s = t.slots;
        for (; k < n; k++) {
            int e = ends[k];
            int length = e - cursor;
            if (length == 1) {
                ids[size++] = singleByte[text[cursor] & 0xFF];
            } else {
                int hit =
                        length <= MAX_BYTES
                                ? lookup(text, cursor, e, ids, size, mask, tags, s, p)
                                : 0;
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
     * Records the word of the probe's last lookup as the {@code count} ids at {@code ids[first]}.
     */
    void insert(int[] ids, int first, int count, Probe p) {
        Table t = p.table;
        int slot = p.slot;
        if (slot < 0) {
            // under the cap, a sixty-fourth of the slots' worth of words without room is too many;
            // at the cap, a table's worth: then the text has moved on from what the table holds
            int slots = t.mask + 1;
            if (++t.homeless > (slots < maxSlots ? slots >>> 6 : slots)) {
                replace(t);
            }
            return;
        }
        if (!TAG.compareAndSet(t.tags, slot, EMPTY, BUSY)) {
            return; // another thread took the slot first
        }
        long[] s = t.slots;
        int at = 4 * slot + 2;
        s[at] = p.lo;
        s[at + 1] = p.hi | ((long) count << 60);
        s[at + 2] = (ids[first] & 0xFFFFFFFFL) | (count > 1 ? (long) ids[first + 1] << 32 : 0);
        s[at + 3] =
                count > 2
                        ? (ids[first + 2] & 0xFFFFFFFFL)
                                | (count > 3 ? (long) ids[first + 3] << 32 : 0)
                        : 0;
        TAG.setRelease(t.tags, slot, p.tag);
    }

    /**
     * A table instead of {@code full}: twice its size with its words while under the cap, else an
     * empty one. One thread replaces at a time; the others go on with the table they have.
     */
    private void replace(Table full) {
        int slots = full.mask + 1;
        boolean grow = slots < maxSlots;
        if (!replacing.compareAndSet(false, true)) {
            return;
        }
        try {
            if (table != full) {
                return;
            }
            if (!grow) {
                table = new Table(slots);
                return;
            }
            Table next = new Table(slots << 1);
            for (int from = 0; from < slots + WINDOW; from++) {
                byte tag = (byte) TAG.getAcquire(full.tags, from);
                if ((tag & 1) == 0) {
                    continue; // empty or busy
                }
                int at = 4 * from + 2;
                long lo = full.slots[at];
                long hi = full.slots[at + 1] & ~COUNT_MASK;
                int home = (int) hash(lo, hi) & next.mask;
                long empty = zeroBytes((long) LONGS.get(next.tags, home));
                if (empty == 0) {
                    continue;
                }
                int to = home + (Long.numberOfTrailingZeros(empty) >>> 3);
                System.arraycopy(full.slots, at, next.slots, 4 * to + 2, 4);
                next.tags[to] = tag;
            }
            table = next; // a volatile write: the copy is out before the table is
        } finally {
            replacing.set(false);
        }
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
