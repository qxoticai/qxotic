package com.qxotic.jam.vector;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class VectorSupportTileTest {

    @Test
    void namedTilesMapToTheirCodes() {
        assertEquals(0, VectorSupport.tileCode("3x2"));
        assertEquals(1, VectorSupport.tileCode("3x4"));
        assertEquals(2, VectorSupport.tileCode("4x4"));
        assertEquals(12, VectorSupport.tileCode("scalar"));
    }

    @Test
    void unknownTileIsRejectedNamingThePropertyAndChoices() {
        IllegalArgumentException e =
                assertThrows(IllegalArgumentException.class, () -> VectorSupport.tileCode("4x3"));
        assertTrue(e.getMessage().startsWith("jam.vector.tile=4x3"), e.getMessage());
        assertTrue(e.getMessage().contains("3x4"), e.getMessage());
    }

    @Test
    void theWideBandNeedsGraalCe25Point4OrNewer() {
        assertTrue(VectorSupport.wideBand(true, "GraalVM CE 25.4.4.1.1+1.1"));
        assertTrue(VectorSupport.wideBand(true, "GraalVM CE 26.0.1+3.1"));
        assertFalse(VectorSupport.wideBand(true, "GraalVM CE 25.2.4+7.1"));
        assertFalse(VectorSupport.wideBand(true, "Oracle GraalVM 25.2.4+7.1"));
        assertFalse(VectorSupport.wideBand(false, "GraalVM CE 25.4.4.1.1+1.1"), "C2 on a CE build");
        assertFalse(VectorSupport.wideBand(false, ""), "OpenJDK C2");
    }
}
