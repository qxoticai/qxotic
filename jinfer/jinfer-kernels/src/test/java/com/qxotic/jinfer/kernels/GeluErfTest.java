package com.qxotic.jinfer.kernels;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

final class GeluErfTest {

    @Test
    void erfIsLibmsToTheLastBit() {
        // erff() of glibc at these points
        assertEquals(0f, GeluErf.erf(0f));
        assertEquals(0.5204998778f, GeluErf.erf(0.5f));
        assertEquals(0.8427007929f, GeluErf.erf(1f));
        assertEquals(-0.9953222650f, GeluErf.erf(-2f));
        assertEquals(0.9999999846f, GeluErf.erf(3.9f), 1e-7f);
        assertEquals(1f, GeluErf.erf(6f));
    }
}
