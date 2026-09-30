package com.qxotic.toknroll.impl;

import java.util.function.LongConsumer;

/**
 * Immutable map from {@code long} keys (packed {@link IntPair}) to {@code long} values, built for
 * the merge table of a BPE model.
 *
 * <p>Two stores. Pairs whose halves are both under {@link #GRID} live in a dense grid: one index
 * load and one value load, no hashing. Token ids follow merge rank closely in a BPE vocabulary, so
 * the grid holds the merges between the alphabet and the earliest products, which is where three
 * quarters of the lookups of a merge loop land. Every other pair goes to an open-addressing table
 * with the key and the value side by side, so a probe touches one cache line: in one long when the
 * ids are small and the merged id is the rank plus a constant, as in every tiktoken vocabulary and
 * most others, else in two.
 */
final class LongLongMap {

    /** Pairs with both halves under this take the grid. */
    private static final int GRID = 512;

    private static final char NO_GRID_ENTRY = Character.MAX_VALUE;
    private static final long EMPTY = 0L;

    // The one-long entry: the pair's halves in 18 bits each, the merged id above them, all
    // biased by one so that zero stays empty.
    private static final int HALF_BITS = 18;
    private static final int HALF_MASK = (1 << HALF_BITS) - 1;
    private static final long KEY_MASK = (1L << (2 * HALF_BITS)) - 1;

    private final char[] gridIndex; // [left << 9 | right] -> index into gridValues
    private final long[] gridValues;
    private final long[] table; // biased key, value, biased key, value, ... or one long per entry
    private final int mask;
    private final int maxProbe;
    private final boolean compact; // one long per entry
    private final int rankOffset; // compact: the merged id minus its rank

    /** Bulk-constructs the map from parallel key/value arrays. */
    LongLongMap(long[] keys, long[] values) {
        if (keys.length != values.length) {
            throw new IllegalArgumentException("keys and values must have the same length");
        }
        int gridEntries = 0;
        for (long key : keys) {
            if (inGrid(key)) {
                gridEntries++;
            }
        }
        if (gridEntries >= NO_GRID_ENTRY) {
            throw new IllegalStateException(gridEntries + " grid entries exceed a char index");
        }
        this.gridIndex = new char[GRID * GRID];
        java.util.Arrays.fill(gridIndex, NO_GRID_ENTRY);
        this.gridValues = new long[gridEntries];

        int capacity = tableSizeFor(keys.length - gridEntries);
        this.mask = capacity - 1;
        this.rankOffset = rankOffset(keys, values);
        this.compact = rankOffset != Integer.MIN_VALUE;
        this.table = new long[compact ? capacity : capacity << 1];
        int nextGridEntry = 0;
        for (int i = 0; i < keys.length; i++) {
            long key = keys[i];
            long value = values[i];
            if (inGrid(key)) {
                gridIndex[gridSlot(key)] = (char) nextGridEntry;
                gridValues[nextGridEntry++] = value;
            } else if (compact) {
                insertCompact(key, value);
            } else {
                insertWide(key, value);
            }
        }
        // Swaps can push an entry further than any insertion saw, so measure at the end.
        int observedMaxProbe = 0;
        for (int slot = 0; slot <= mask; slot++) {
            long stored = compact ? table[slot] : table[slot << 1];
            if (stored != EMPTY) {
                long key = compact ? (stored - 1) & KEY_MASK : stored - 1;
                int probe = (slot - (hash(key) & mask)) & mask;
                observedMaxProbe = Math.max(observedMaxProbe, probe);
            }
        }
        this.maxProbe = observedMaxProbe;
    }

    /**
     * The constant every merged id exceeds its rank by, when there is one and every half of every
     * key and every merged id fits the one-long entry; else {@link Integer#MIN_VALUE}.
     */
    private static int rankOffset(long[] keys, long[] values) {
        if (keys.length == 0) {
            return 0;
        }
        int offset = IntPair.left(values[0]) - IntPair.right(values[0]);
        for (int i = 0; i < keys.length; i++) {
            long key = keys[i];
            if (inGrid(key)) {
                continue;
            }
            int left = IntPair.left(key);
            int right = IntPair.right(key);
            int id = IntPair.left(values[i]);
            int rank = IntPair.right(values[i]);
            if ((left | right) < 0
                    || (left | right) > HALF_MASK
                    || id < 0
                    || id - rank != offset
                    || (long) id + 1 >= 1L << (64 - 2 * HALF_BITS)) {
                return Integer.MIN_VALUE;
            }
        }
        return offset;
    }

    /** A one-long entry: the halves of the key, the merged id above them, plus one. */
    private static long compactEntry(long key, int id) {
        long halves = ((long) IntPair.left(key) << HALF_BITS) | IntPair.right(key);
        return (halves | ((long) id << (2 * HALF_BITS))) + 1;
    }

    /** The key of a one-long entry, as the halves are packed there: not the raw pair. */
    private static long compactKey(int left, int right) {
        return ((long) left << HALF_BITS) | right;
    }

    private void insertCompact(long rawKey, long value) {
        long entry = compactEntry(rawKey, IntPair.left(value));
        long key = (entry - 1) & KEY_MASK;
        int slot = hash(key) & mask;
        int probe = 0;
        while (true) {
            long stored = table[slot];
            if (stored == EMPTY) {
                table[slot] = entry;
                return;
            }
            // Robin Hood: the entry further from home keeps the slot.
            long storedKey = (stored - 1) & KEY_MASK;
            int residentProbe = (slot - (hash(storedKey) & mask)) & mask;
            if (residentProbe < probe) {
                table[slot] = entry;
                entry = stored;
                key = storedKey;
                probe = residentProbe;
            }
            slot = (slot + 1) & mask;
            probe++;
        }
    }

    private void insertWide(long rawKey, long value) {
        long biasedKey = rawKey + 1;
        int slot = hash(rawKey) & mask;
        int probe = 0;
        while (true) {
            long stored = table[slot << 1];
            if (stored == EMPTY) {
                table[slot << 1] = biasedKey;
                table[(slot << 1) + 1] = value;
                return;
            }
            int residentProbe = (slot - (hash(stored - 1) & mask)) & mask;
            if (residentProbe < probe) {
                long tmpKey = stored;
                long tmpValue = table[(slot << 1) + 1];
                table[slot << 1] = biasedKey;
                table[(slot << 1) + 1] = value;
                biasedKey = tmpKey;
                value = tmpValue;
                probe = residentProbe;
            }
            slot = (slot + 1) & mask;
            probe++;
        }
    }

    private static boolean inGrid(long key) {
        return (IntPair.left(key) | IntPair.right(key)) >= 0
                && (IntPair.left(key) | IntPair.right(key)) < GRID;
    }

    private static int gridSlot(long key) {
        return (IntPair.left(key) << 9) | IntPair.right(key);
    }

    /**
     * Returns the value associated with {@code key}, or {@link IntPair#NONE} ({@code -1L}) if
     * absent.
     */
    final long get(long key) {
        return getPair(IntPair.left(key), IntPair.right(key));
    }

    /** Convenience overload for packed int pair keys. */
    final long getPair(int left, int right) {
        if ((left | right) < GRID && (left | right) >= 0) {
            int entry = gridIndex[(left << 9) | right];
            return entry == NO_GRID_ENTRY ? IntPair.NONE : gridValues[entry];
        }
        if (compact) {
            if ((left | right) < 0 || (left | right) > HALF_MASK) {
                return IntPair.NONE;
            }
            return getCompact(compactKey(left, right));
        }
        return getWide(((long) left << 32) | (right & 0xFFFFFFFFL));
    }

    private long getCompact(long key) {
        long[] t = table;
        int m = mask;
        int slot = hash(key) & m;
        for (int remaining = maxProbe; ; remaining--) {
            long stored = t[slot];
            if (((stored - 1) & KEY_MASK) == key && stored != EMPTY) {
                int id = (int) ((stored - 1) >>> (2 * HALF_BITS));
                return IntPair.of(id, id - rankOffset);
            }
            if (stored == EMPTY || remaining == 0) {
                return IntPair.NONE;
            }
            slot = (slot + 1) & m;
        }
    }

    private long getWide(long key) {
        long biasedKey = key + 1;
        long[] t = table;
        int m = mask;
        int slot = hash(key) & m;
        for (int remaining = maxProbe; ; remaining--) {
            long stored = t[slot << 1];
            if (stored == biasedKey) {
                return t[(slot << 1) + 1];
            }
            if (stored == EMPTY || remaining == 0) {
                return IntPair.NONE;
            }
            slot = (slot + 1) & m;
        }
    }

    final void forEachValue(LongConsumer consumer) {
        for (long value : gridValues) {
            consumer.accept(value);
        }
        for (int slot = 0; slot <= mask; slot++) {
            if (compact) {
                long stored = table[slot];
                if (stored != EMPTY) {
                    int id = (int) ((stored - 1) >>> (2 * HALF_BITS));
                    consumer.accept(IntPair.of(id, id - rankOffset));
                }
            } else if (table[slot << 1] != EMPTY) {
                consumer.accept(table[(slot << 1) + 1]);
            }
        }
    }

    /** Murmur3 fmix64. */
    private static int hash(long key) {
        key ^= key >>> 33;
        key *= 0xff51afd7ed558ccdL;
        key ^= key >>> 33;
        key *= 0xc4ceb9fe1a85ec53L;
        key ^= key >>> 33;
        return (int) key;
    }

    /** Returns a power-of-two capacity tuned for low probe variance. */
    private static int tableSizeFor(int entries) {
        int minCapacity = Math.max(4, (entries << 1) + 1);
        return Integer.highestOneBit(minCapacity - 1) << 1;
    }
}
