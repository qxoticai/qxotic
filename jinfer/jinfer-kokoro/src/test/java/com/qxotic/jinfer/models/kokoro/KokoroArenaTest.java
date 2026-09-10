package com.qxotic.jinfer.models.kokoro;

import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.memory.MemoryAllocators;
import com.qxotic.jota.memory.MemoryArena;
import java.lang.foreign.Arena;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import org.junit.jupiter.api.Test;

class KokoroArenaTest {
    // Exercise state construction without requiring a downloaded model and voice pack.
    private static Constructor<Kokoro.State> constructor() throws Exception {
        var constructor =
                Kokoro.State.class.getDeclaredConstructor(MemoryArena.class, MemoryArena.class);
        constructor.setAccessible(true);
        return constructor;
    }

    @Test
    void speechStateRefusesConfinedMemoryWithoutClosingIt() throws Exception {
        var constructor = constructor();
        try (Arena arena = Arena.ofConfined()) {
            var failure =
                    assertThrows(
                            InvocationTargetException.class,
                            () -> constructor.newInstance(MemoryAllocators.ofArena(arena), null));
            var cause = assertInstanceOf(IllegalArgumentException.class, failure.getCause());
            assertTrue(cause.getMessage().contains("Confined arenas"));
            assertTrue(arena.scope().isAlive());
        }
    }

    @Test
    void speechStateBorrowsSharedMemory() throws Exception {
        try (Arena arena = Arena.ofShared()) {
            constructor().newInstance(MemoryAllocators.ofArena(arena), null).close();
            assertTrue(arena.scope().isAlive());
        }
    }
}
