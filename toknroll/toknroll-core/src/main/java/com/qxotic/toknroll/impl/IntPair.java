package com.qxotic.toknroll.impl;

/** Packs two ints into a single long. Sentinel value: {@code -1L}. */
final class IntPair {
    /** Sentinel indicating no pair. */
    static final long NONE = -1L;

    private IntPair() {}

    /** Packs {@code left} into the high 32 bits and {@code right} into the low 32 bits. */
    static long of(int left, int right) {
        return ((long) left << 32) | (right & 0xFFFFFFFFL);
    }

    /** Extracts the high 32 bits (left half) of a packed pair. */
    static int left(long pair) {
        return (int) (pair >>> 32);
    }

    /** Extracts the low 32 bits (right half) of a packed pair. */
    static int right(long pair) {
        return (int) pair;
    }
}
