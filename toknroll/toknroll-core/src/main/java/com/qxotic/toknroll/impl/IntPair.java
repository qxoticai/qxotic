package com.qxotic.toknroll.impl;

/** Packs two ints into a single long. Sentinel value: {@code -1L}. */
public final class IntPair {

    public static final long NONE = -1L;

    private final int first;
    private final int second;

    public IntPair(int first, int second) {
        this.first = first;
        this.second = second;
    }

    public int first() {
        return first;
    }

    public int second() {
        return second;
    }

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
