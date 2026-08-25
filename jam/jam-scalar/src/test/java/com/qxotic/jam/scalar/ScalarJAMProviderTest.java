package com.qxotic.jam.scalar;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;

import com.qxotic.jam.JAM;
import org.junit.jupiter.api.Test;

class ScalarJAMProviderTest {

    @Test
    void scalarProviderIsDiscoverable() {
        JAM.Provider provider = JAM.providers().getFirst();

        assertEquals("scalar", provider.id());
        assertEquals(0, provider.priority());
        assertInstanceOf(ScalarJAM.class, provider.create());
    }
}
