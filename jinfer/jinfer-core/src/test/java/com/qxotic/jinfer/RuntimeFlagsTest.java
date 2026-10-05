package com.qxotic.jinfer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class RuntimeFlagsTest {

    @Test
    void positiveIntParsesOrFallsBackToTheDefault() {
        assertEquals(512, RuntimeFlags.positiveInt("jinfer.decodeBlockSize", null, 512));
        assertEquals(64, RuntimeFlags.positiveInt("jinfer.decodeBlockSize", "64", 512));
        assertEquals(64, RuntimeFlags.positiveInt("jinfer.decodeBlockSize", " 64 ", 512));
    }

    @Test
    void zeroNegativeAndGarbageFailNamingTheProperty() {
        for (String bad : new String[] {"0", "-4", "abc", "", "1.5"}) {
            IllegalArgumentException e =
                    assertThrows(
                            IllegalArgumentException.class,
                            () -> RuntimeFlags.positiveInt("jinfer.decodeBlockSize", bad, 512),
                            bad);
            assertTrue(e.getMessage().startsWith("jinfer.decodeBlockSize must be"), e.getMessage());
        }
    }
}
