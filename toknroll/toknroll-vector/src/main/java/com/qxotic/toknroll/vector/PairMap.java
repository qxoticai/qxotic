package com.qxotic.toknroll.vector;

import com.qxotic.toknroll.spi.TiktokenTables;
import java.util.Arrays;

/**
 * The merges of a model: a pair of token ids to the token it merges into and its rank, packed as
 * {@code id << 32 | rank}, or {@link #NONE}.
 *
 * <p>Two stores. Pairs whose ids are both under {@link #GRID} live in a dense grid: one index load
 * and one value load, no hashing. Token ids follow merge rank closely in a BPE vocabulary, so the
 * grid holds the merges between the alphabet and the earliest products, where three quarters of the
 * lookups of a merge loop land. Every other pair goes to an open-addressing table with the key and
 * the value side by side, so a probe touches one cache line: in one long when the ids are small and
 * the merged id is the rank plus a constant, as in every tiktoken vocabulary, else in two.
 */
final class PairMap {
    static final long NONE = -1L;

    /** Pairs with both ids under this take the grid. */
    private static final int GRID = 512;

    private static final char NO_GRID_ENTRY = Character.MAX_VALUE;
    private static final long EMPTY = 0L;

    // The one-long entry: the pair's ids in 18 bits each, the merged id above them, all biased by
    // one so that zero stays empty.
    private static final int HALF_BITS = 18;
    private static final int HALF_MASK = (1 << HALF_BITS) - 1;
    private static final long KEY_MASK = (1L << (2 * HALF_BITS)) - 1;

    private final char[] gridIndex; // [left << 9 | right] -> index into gridValues
    private final long[] gridValues;
    private final long[] table; // biased key, value, ... or one long per entry
    private final int mask;
    private final int maxProbe;
    private final boolean compact; // one long per entry
    private final int rankOffset; // compact: the merged id minus its rank

    PairMap(TiktokenTables model) {
        int count = model.mergeCount();
        long[] keys = new long[count];
        long[] values = new long[count];
        for (int i = 0; i < count; i++) {
            keys[i] = pack(model.mergeLeft(i), model.mergeRight(i));
            values[i] = pack(model.mergeResult(i), model.mergeRank(i));
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
        Arrays.fill(gridIndex, NO_GRID_ENTRY);
        this.gridValues = new long[gridEntries];

        int capacity = tableSizeFor(count - gridEntries);
        this.mask = capacity - 1;
        this.rankOffset = rankOffset(keys, values);
        this.compact = rankOffset != Integer.MIN_VALUE;
        this.table = new long[compact ? capacity : capacity << 1];
        int nextGridEntry = 0;
        for (int i = 0; i < count; i++) {
            if (inGrid(keys[i])) {
                gridIndex[gridSlot(keys[i])] = (char) nextGridEntry;
                gridValues[nextGridEntry++] = values[i];
            } else if (compact) {
                insertCompact(keys[i], values[i]);
            } else {
                insertWide(keys[i], values[i]);
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

    private static long pack(int high, int low) {
        return ((long) high << 32) | (low & 0xFFFFFFFFL);
    }

    private static int high(long pair) {
        return (int) (pair >>> 32);
    }

    private static int low(long pair) {
        return (int) pair;
    }

    /**
     * The constant every merged id exceeds its rank by, when there is one and every id of every key
     * and every merged id fits the one-long entry; else {@link Integer#MIN_VALUE}.
     */
    private static int rankOffset(long[] keys, long[] values) {
        if (keys.length == 0) {
            return 0;
        }
        int offset = high(values[0]) - low(values[0]);
        for (int i = 0; i < keys.length; i++) {
            long key = keys[i];
            if (inGrid(key)) {
                continue;
            }
            int left = high(key);
            int right = low(key);
            int id = high(values[i]);
            int rank = low(values[i]);
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

    /** A one-long entry: the ids of the key, the merged id above them, plus one. */
    private static long compactEntry(long key, int id) {
        long halves = ((long) high(key) << HALF_BITS) | low(key);
        return (halves | ((long) id << (2 * HALF_BITS))) + 1;
    }

    /** The key of a one-long entry, as the ids are packed there: not the raw pair. */
    private static long compactKey(int left, int right) {
        return ((long) left << HALF_BITS) | right;
    }

    private void insertCompact(long rawKey, long value) {
        long entry = compactEntry(rawKey, high(value));
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
        return (high(key) | low(key)) >= 0 && (high(key) | low(key)) < GRID;
    }

    private static int gridSlot(long key) {
        return (high(key) << 9) | low(key);
    }

    /** The merge of {@code left} and {@code right}: {@code id << 32 | rank}, or {@link #NONE}. */
    long get(int left, int right) {
        if ((left | right) < GRID && (left | right) >= 0) {
            int entry = gridIndex[(left << 9) | right];
            return entry == NO_GRID_ENTRY ? NONE : gridValues[entry];
        }
        if (compact) {
            if ((left | right) < 0 || (left | right) > HALF_MASK) {
                return NONE;
            }
            return getCompact(compactKey(left, right));
        }
        return getWide(pack(left, right));
    }

    private long getCompact(long key) {
        long[] t = table;
        int m = mask;
        int slot = hash(key) & m;
        for (int remaining = maxProbe; ; remaining--) {
            long stored = t[slot];
            if (((stored - 1) & KEY_MASK) == key && stored != EMPTY) {
                int id = (int) ((stored - 1) >>> (2 * HALF_BITS));
                return pack(id, id - rankOffset);
            }
            if (stored == EMPTY || remaining == 0) {
                return NONE;
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
                return NONE;
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

    /** A power-of-two capacity at least twice the entries, for short probes. */
    private static int tableSizeFor(int entries) {
        int minCapacity = Math.max(4, (entries << 1) + 1);
        return Integer.highestOneBit(minCapacity - 1) << 1;
    }
}
