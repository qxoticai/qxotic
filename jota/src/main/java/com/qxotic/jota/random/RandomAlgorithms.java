package com.qxotic.jota.random;

public final class RandomAlgorithms {
    private RandomAlgorithms() {}

    public static float uniformFp32(long elementIndex, RandomKey key) {
        long key0 = key0(key);
        long key1 = key1(key);
        return uniformFp32(elementIndex, key0, key1);
    }

    public static float uniformFp32(long elementIndex, long key0, long key1) {
        long state0 = nextState(seed(elementIndex, key0, key1));
        long u24 = modPositive(state0, 1L << 24);
        return (float) u24 / 16777216.0f;
    }

    public static double uniformFp64(long elementIndex, RandomKey key) {
        long key0 = key0(key);
        long key1 = key1(key);
        return uniformFp64(elementIndex, key0, key1);
    }

    public static double uniformFp64(long elementIndex, long key0, long key1) {
        long state0 = nextState(seed(elementIndex, key0, key1));
        long state1 = nextState(state0);
        long hi26 = modPositive(state0, 1L << 26);
        long lo27 = modPositive(state1, 1L << 27);
        long bits53 = hi26 * (1L << 27) + lo27;
        return bits53 * 0x1.0p-53;
    }

    public static long key0(RandomKey key) {
        return lcgKey(key).k0();
    }

    public static long key1(RandomKey key) {
        return lcgKey(key).k1();
    }

    private static LcgRandomKey lcgKey(RandomKey key) {
        if (key instanceof LcgRandomKey lcg) {
            return lcg;
        }
        throw new IllegalArgumentException(
                "Unsupported key implementation/tag for LCG random algorithm: "
                        + key.algorithmTag());
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
