package com.qxotic.toknroll.impl;

/**
 * Immutable map from {@code long} keys (packed {@link IntPair}) to {@code long} values, built for
 * the merge table of a BPE model.
 *
 * <p>Two stores. Pairs whose halves are both under {@link #GRID} live in a dense grid: one index
 * load and one value load, no hashing. Token ids follow merge rank closely in a BPE vocabulary, so
 * the grid holds the merges between the alphabet and the earliest products, which is where three
 * quarters of the lookups of a merge loop land. Every other pair goes to an open-addressing table
 * with the key and the value side by side, so a probe touches one cache line.
 */
final class LongLongMap {

    /** Pairs with both halves under this take the grid. */
    private static final int GRID = 512;

    private static final char NO_GRID_ENTRY = Character.MAX_VALUE;
    private static final long EMPTY = 0L;

    private final char[] gridIndex; // [left << 9 | right] -> index into gridValues
    private final long[] gridValues;
    private final long[] table; // biased key, value, biased key, value, ...
    private final int mask;
    private final int maxProbe;

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
        this.table = new long[capacity << 1];
        int observedMaxProbe = 0;
        int nextGridEntry = 0;
        for (int i = 0; i < keys.length; i++) {
            long currentRawKey = keys[i];
            long currentValue = values[i];
            if (inGrid(currentRawKey)) {
                gridIndex[gridSlot(currentRawKey)] = (char) nextGridEntry;
                gridValues[nextGridEntry++] = currentValue;
                continue;
            }
            long currentBiasedKey = currentRawKey + 1;
            int slot = hash(currentRawKey) & mask;
            int probe = 0;
            while (true) {
                long stored = table[slot << 1];
                if (stored == EMPTY) {
                    table[slot << 1] = currentBiasedKey;
                    table[(slot << 1) + 1] = currentValue;
                    break;
                }
                // Robin Hood: the entry further from home keeps the slot.
                int residentProbe = (slot - (hash(stored - 1) & mask)) & mask;
                if (residentProbe < probe) {
                    long tmpKey = stored;
                    long tmpValue = table[(slot << 1) + 1];
                    table[slot << 1] = currentBiasedKey;
                    table[(slot << 1) + 1] = currentValue;
                    currentBiasedKey = tmpKey;
                    currentValue = tmpValue;
                    probe = residentProbe;
                }
                slot = (slot + 1) & mask;
                probe++;
            }
        }
        // Swaps can push an entry further than any insertion saw, so measure at the end.
        for (int slot = 0; slot <= mask; slot++) {
            long stored = table[slot << 1];
            if (stored != EMPTY) {
                int probe = (slot - (hash(stored - 1) & mask)) & mask;
                observedMaxProbe = Math.max(observedMaxProbe, probe);
            }
        }
        this.maxProbe = observedMaxProbe;
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
        if (inGrid(key)) {
            int entry = gridIndex[gridSlot(key)];
            return entry == NO_GRID_ENTRY ? IntPair.NONE : gridValues[entry];
        }
        return getHashed(key);
    }

    /** Convenience overload for packed int pair keys. */
    final long getPair(int left, int right) {
        if ((left | right) < GRID && (left | right) >= 0) {
            int entry = gridIndex[(left << 9) | right];
            return entry == NO_GRID_ENTRY ? IntPair.NONE : gridValues[entry];
        }
        return getHashed(((long) left << 32) | (right & 0xFFFFFFFFL));
    }

    private long getHashed(long key) {
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

    /** Every entry, in no particular order. */
    final void forEach(EntryConsumer consumer) {
        for (int slot = 0; slot < gridIndex.length; slot++) {
            if (gridIndex[slot] != NO_GRID_ENTRY) {
                consumer.accept(
                        IntPair.of(slot >>> 9, slot & (GRID - 1)), gridValues[gridIndex[slot]]);
            }
        }
        for (int slot = 0; slot <= mask; slot++) {
            long stored = table[slot << 1];
            if (stored != EMPTY) {
                consumer.accept(stored - 1, table[(slot << 1) + 1]);
            }
        }
    }

    interface EntryConsumer {
        void accept(long key, long value);
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
