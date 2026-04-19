package com.qxotic.toknroll.impl;

/**
 * Immutable open-addressing hash map from {@code long} keys (packed {@link IntPair}) to {@code
 * long} values. Backed by flat primitive arrays with no object overhead.
 */
final class LongLongMap {

    private static final long EMPTY = 0L;

    private final long[] table;
    private final int capacity;
    private final int mask;
    private final int maxProbe;

    /** Bulk-constructs the map from parallel key/value arrays. */
    LongLongMap(long[] keys, long[] values) {
        if (keys.length != values.length) {
            throw new IllegalArgumentException("keys and values must have the same length");
        }
        this.capacity = tableSizeFor(keys.length);
        this.mask = capacity - 1;
        this.table = new long[capacity << 1];

        int observedMaxProbe = 0;

        for (int i = 0; i < keys.length; i++) {
            long currentRawKey = keys[i];
            long currentBiasedKey = currentRawKey + 1;
            long currentValue = values[i];
            int slot = hash(currentRawKey) & mask;
            int probe = 0;

            while (true) {
                long stored = table[slot];
                if (stored == EMPTY) {
                    table[slot] = currentBiasedKey;
                    table[capacity + slot] = currentValue;
                    if (probe > observedMaxProbe) {
                        observedMaxProbe = probe;
                    }
                    break;
                }

                int residentProbe = (slot - (hash(stored - 1) & mask)) & mask;
                if (residentProbe < probe) {
                    long tmpKey = stored;
                    long tmpValue = table[capacity + slot];
                    table[slot] = currentBiasedKey;
                    table[capacity + slot] = currentValue;
                    currentBiasedKey = tmpKey;
                    currentValue = tmpValue;
                    probe = residentProbe;
                }

                slot = (slot + 1) & mask;
                probe++;
            }
        }

        this.maxProbe = observedMaxProbe;
    }

    /**
     * Returns the value associated with {@code key}, or {@link IntPair#NONE} ({@code -1L}) if
     * absent.
     */
    final long get(long key) {
        long biasedKey = key + 1;
        long[] t = table;
        int c = capacity;
        int m = mask;
        int slot = hash(key) & m;
        int remaining = maxProbe;
        while (true) {
            long stored = t[slot];
            if (stored == EMPTY) {
                return IntPair.NONE;
            }
            if (stored == biasedKey) {
                return t[c + slot];
            }
            if (remaining-- == 0) {
                return IntPair.NONE;
            }

            slot = (slot + 1) & m;
            stored = t[slot];
            if (stored == EMPTY) {
                return IntPair.NONE;
            }
            if (stored == biasedKey) {
                return t[c + slot];
            }
            if (remaining-- == 0) {
                return IntPair.NONE;
            }

            slot = (slot + 1) & m;
        }
    }

    /** Convenience overload for packed int pair keys. */
    final long getPair(int left, int right) {
        long key = ((long) left << 32) | (right & 0xFFFFFFFFL);
        long biasedKey = key + 1;
        long[] t = table;
        int c = capacity;
        int m = mask;
        int slot = hash(key) & m;
        int remaining = maxProbe;
        while (true) {
            long stored = t[slot];
            if (stored == EMPTY) {
                return IntPair.NONE;
            }
            if (stored == biasedKey) {
                return t[c + slot];
            }
            if (remaining-- == 0) {
                return IntPair.NONE;
            }

            slot = (slot + 1) & m;
            stored = t[slot];
            if (stored == EMPTY) {
                return IntPair.NONE;
            }
            if (stored == biasedKey) {
                return t[c + slot];
            }
            if (remaining-- == 0) {
                return IntPair.NONE;
            }

            slot = (slot + 1) & m;
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
