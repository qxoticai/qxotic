package com.qxotic.jinfer.models.inflect2;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.memory.MemoryAllocators;
import java.lang.foreign.Arena;
import org.junit.jupiter.api.Test;

class Inflect2ArenaTest {
    @Test
    void speechStateDiagnosesConfinedMemoryWithoutClosingIt() {
        try (Arena arena = Arena.ofConfined()) {
            var failure =
                    assertThrows(
                            AssertionError.class,
                            () -> new Inflect2.State(MemoryAllocators.ofArena(arena), null));
            assertTrue(failure.getMessage().contains("Confined arenas"));
            assertTrue(arena.scope().isAlive());
        }
    }

    @Test
    void speechStateBorrowsSharedMemory() {
        try (Arena arena = Arena.ofShared()) {
            new Inflect2.State(MemoryAllocators.ofArena(arena), null).close();
            assertTrue(arena.scope().isAlive());
        }
    }
}
