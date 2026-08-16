package com.qxotic.jinfer.boundary;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jinfer.PanamaMemoryArena;
import java.lang.foreign.Arena;
import org.junit.jupiter.api.Test;

/** Borrowed-memory liveness is checked before any operation reaches a kernel. */
class SafetyCanaryTest {

    @Test
    void closedBorrowedArenaFailsWithTheBoundaryMessage() {
        Arena arena = Arena.ofShared();
        RuntimeStateLifecycleTest.ProbeState state =
                new RuntimeStateLifecycleTest.ProbeState(new PanamaMemoryArena(arena), false);
        state.exclusively(() -> {});
        arena.close();

        IllegalStateException failure =
                assertThrows(IllegalStateException.class, () -> state.exclusively(() -> {}));
        assertEquals(ContextState.FREED_MESSAGE, failure.getMessage());
        assertThrows(IllegalStateException.class, () -> state.exclusively(() -> {}));
    }

    @Test
    void undyingScopesRemainUsable() {
        new RuntimeStateLifecycleTest.ProbeState(new PanamaMemoryArena(Arena.ofAuto()), false)
                .exclusively(() -> {});
        new RuntimeStateLifecycleTest.ProbeState(new PanamaMemoryArena(Arena.global()), false)
                .exclusively(() -> {});
    }
}
