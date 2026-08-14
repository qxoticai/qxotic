package com.qxotic.jinfer.x.server;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

import com.qxotic.jinfer.x.llm.Sampling;
import java.util.Map;
import org.junit.jupiter.api.Test;

class GenerationTest {

    @Test
    void anOmittedSeedStaysUnseeded() {
        Sampling defaults = new Sampling(0.8f, 0.95f, 40, 0.05f, null);
        assertNull(Generation.sampling(Map.of(), defaults).seed());
        assertEquals(42L, Generation.sampling(Map.of("seed", 42), defaults).seed());
    }
}
