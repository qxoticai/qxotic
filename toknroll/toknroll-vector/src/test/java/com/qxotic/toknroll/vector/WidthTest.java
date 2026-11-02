package com.qxotic.toknroll.vector;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class WidthTest {

    /** The build runs every test once per width: this run's is the one it asked for. */
    @Test
    void theRunUsesTheRequestedWidth() {
        String requested = System.getProperty("toknroll.vector.width");
        if (requested != null) {
            assertEquals(Integer.parseInt(requested), Width.BITS);
        }
    }

    @Test
    void theDefaultIsBetween128And512() {
        int bits = Width.bits(null);
        assertTrue(bits == 128 || bits == 256 || bits == 512, "default " + bits);
        assertEquals(bits, Width.bits(" "));
    }

    @Test
    void onlyTheThreeWidthsAreTaken() {
        assertEquals(128, Width.bits("128"));
        assertEquals(256, Width.bits(" 256 "));
        assertEquals(512, Width.bits("512"));
        assertThrows(IllegalArgumentException.class, () -> Width.bits("64"));
        assertThrows(IllegalArgumentException.class, () -> Width.bits("1024"));
        assertThrows(NumberFormatException.class, () -> Width.bits("wide"));
    }
}
