package com.qxotic.jam.vector;

import static org.junit.jupiter.api.Assertions.assertEquals;
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
}
