package com.qxotic.jota.random;

/**
 * Factory for creating opaque random keys from seeds.
 *
 * <p>Most callers should use {@code Tensor.randomKey(seed)}.
 */
public final class RandomKeys {

    private RandomKeys() {}

    public static RandomKey key(long seed) {
        return LcgRandomKey.fromSeed(seed);
    }
}
