package com.qxotic.jota.random;

import java.util.Objects;

final class LcgRandomKey implements RandomKey {
    static final String TAG = "lcg_v1";

    private final long k0;
    private final long k1;

    private LcgRandomKey(long k0, long k1) {
        this.k0 = k0;
        this.k1 = k1;
    }

    static LcgRandomKey fromSeed(long seed) {
        long a = mix64(seed ^ 0x243f6a8885a308d3L);
        long b = mix64(seed ^ 0x13198a2e03707344L);
        return new LcgRandomKey(a, b);
    }

    @Override
    public String algorithmTag() {
        return TAG;
    }

    @Override
    public RandomKey split(long stream) {
        long a = mix64(k0 ^ stream ^ 0x9e3779b97f4a7c15L);
        long b = mix64(k1 ^ Long.rotateLeft(stream, 17) ^ 0xbf58476d1ce4e5b9L);
        return new LcgRandomKey(a, b);
    }

    @Override
    public RandomKey foldIn(long data) {
        return new LcgRandomKey(mix64(k0 ^ data), mix64(k1 ^ Long.rotateLeft(data, 23)));
    }

    long k0() {
        return k0;
    }

    long k1() {
        return k1;
    }

    private static long mix64(long z) {
        z = (z ^ (z >>> 30)) * 0xbf58476d1ce4e5b9L;
        z = (z ^ (z >>> 27)) * 0x94d049bb133111ebL;
        return z ^ (z >>> 31);
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (!(obj instanceof LcgRandomKey other)) {
            return false;
        }
        return k0 == other.k0 && k1 == other.k1;
    }

    @Override
    public int hashCode() {
        return Objects.hash(k0, k1);
    }

    @Override
    public String toString() {
        return "RandomKey(" + TAG + ")";
    }
}
