package com.qxotic.jota.random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;

import org.junit.jupiter.api.Test;

class RandomKeyTest {

    @Test
    void ofIsDeterministicForSameSeed() {
        RandomKey a = RandomKey.of(1234L);
        RandomKey b = RandomKey.of(1234L);
        assertEquals(a, b);
        assertEquals(a.hashCode(), b.hashCode());
    }

    @Test
    void ofProducesDifferentKeysForDifferentSeeds() {
        RandomKey a = RandomKey.of(1L);
        RandomKey b = RandomKey.of(2L);
        assertNotEquals(a, b);
    }

    @Test
    void splitIsDeterministicAndStreamSensitive() {
        RandomKey base = RandomKey.of(42L);
        RandomKey s0a = base.split(0L);
        RandomKey s0b = base.split(0L);
        RandomKey s1 = base.split(1L);

        assertEquals(s0a, s0b);
        assertNotEquals(s0a, s1);
    }

    @Test
    void foldInIsDeterministicAndDataSensitive() {
        RandomKey base = RandomKey.of(99L);
        RandomKey f7a = base.foldIn(7L);
        RandomKey f7b = base.foldIn(7L);
        RandomKey f8 = base.foldIn(8L);

        assertEquals(f7a, f7b);
        assertNotEquals(f7a, f8);
    }

    @Test
    void splitAndFoldInComposeDeterministically() {
        RandomKey base = RandomKey.of(2026L);
        RandomKey a = base.split(5L).foldIn(17L);
        RandomKey b = base.split(5L).foldIn(17L);
        RandomKey c = base.split(5L).foldIn(18L);

        assertEquals(a, b);
        assertNotEquals(a, c);
    }
}
