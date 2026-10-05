package com.qxotic.jinfer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class SegmentsTest {

    @Test
    void vectorBitSizeTakesTheOverrideOrThePreference() {
        assertEquals(256, Segments.vectorBitSize(null, 256));
        assertEquals(0, Segments.vectorBitSize("0", 256));
        assertEquals(128, Segments.vectorBitSize("128", 256));
        assertEquals(512, Segments.vectorBitSize(" 512 ", 256));
    }

    @Test
    void aBadVectorBitSizeFailsNamingTheProperty() {
        for (String bad : new String[] {"abc", "", "100", "-128", "256bits"}) {
            IllegalArgumentException e =
                    assertThrows(
                            IllegalArgumentException.class,
                            () -> Segments.vectorBitSize(bad, 256),
                            bad);
            assertTrue(e.getMessage().startsWith("jinfer.vectorBitSize: "), e.getMessage());
            assertTrue(e.getMessage().endsWith("not '" + bad + "'"), e.getMessage());
        }
    }
}
