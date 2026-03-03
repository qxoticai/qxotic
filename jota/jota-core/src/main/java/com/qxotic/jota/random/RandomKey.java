package com.qxotic.jota.random;

/** Opaque key for deterministic random number generation. */
public interface RandomKey {

    /** Returns the algorithm tag this key is for (e.g., {@code lcg_v1}). */
    String algorithmTag();

    /** Derives an independent deterministic substream key. */
    RandomKey split(long stream);

    /** Mixes external data into the key deterministically. */
    RandomKey foldIn(long data);

    /**
     * Convenience alias for {@link RandomKeys#key(long)}.
     *
     * <p>For user-facing code, prefer {@code Tensor.randomKey(seed)}.
     */
    static RandomKey of(long seed) {
        return RandomKeys.key(seed);
    }
}
