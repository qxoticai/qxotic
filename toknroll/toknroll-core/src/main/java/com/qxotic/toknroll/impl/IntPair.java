package com.qxotic.toknroll.impl;

/** Packs two ints into a single long. Sentinel value: {@code -1L}. */
public final class IntPair {
    public static final long NONE = -1L;

    private IntPair() {}

    public static long of(int left, int right) {
        return ((long) left << 32) | (right & 0xFFFFFFFFL);
    }

    public static int left(long pair) {
        return (int) (pair >>> 32);
    }

    public static int right(long pair) {
        return (int) pair;
    }
}
