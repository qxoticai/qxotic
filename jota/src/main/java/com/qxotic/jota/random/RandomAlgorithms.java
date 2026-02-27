package com.qxotic.jota.random;

public final class RandomAlgorithms {
    private RandomAlgorithms() {}

    public static float uniformFp32(long elementIndex, long key0, long key1) {
        long state0 = nextState(seed(elementIndex, key0, key1));
        long u24 = modPositive(state0, 1L << 24);
        return (float) u24 / 16777216.0f;
    }

    public static double uniformFp64(long elementIndex, long key0, long key1) {
        long state0 = nextState(seed(elementIndex, key0, key1));
        long state1 = nextState(state0);
        long hi26 = modPositive(state0, 1L << 26);
        long lo27 = modPositive(state1, 1L << 27);
        long bits53 = hi26 * (1L << 27) + lo27;
        return bits53 * 0x1.0p-53;
    }

    private static long seed(long elementIndex, long key0, long key1) {
        long k0 = Math.floorMod((int) key0, 1024);
        long k1 = Math.floorMod((int) key1, 1024);
        return elementIndex + (k0 * 1009L) + (k1 * 9176L);
    }

    private static long nextState(long value) {
        return value * 1664525L + 1013904223L;
    }

    private static long modPositive(long value, long bound) {
        long mod = value % bound;
        return ((mod + bound) % bound);
    }
}
