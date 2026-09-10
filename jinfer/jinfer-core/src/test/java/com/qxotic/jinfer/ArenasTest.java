package com.qxotic.jinfer;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.memory.MemoryAllocators;
import java.lang.foreign.Arena;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;

class ArenasTest {

    @Test
    void confinedArenasFailOnTheirOwnerAndOnASingleWorker() throws Exception {
        try (Arena arena = Arena.ofConfined();
                var worker = Executors.newSingleThreadExecutor()) {
            var owner =
                    assertThrows(
                            IllegalArgumentException.class, () -> Arenas.requireCrossThread(arena));
            var other =
                    worker.submit(
                                    () ->
                                            assertThrows(
                                                    IllegalArgumentException.class,
                                                    () -> Arenas.requireCrossThread(arena)))
                            .get(5, TimeUnit.SECONDS);
            assertEquals(owner.getMessage(), other.getMessage());
            assertTrue(owner.getMessage().contains("even with one worker"));
            assertTrue(owner.getMessage().contains("Arenas.newCrossThread()"));
            assertTrue(owner.getMessage().contains("crash the JVM"));
            assertTrue(arena.scope().isAlive());
        }
    }

    @Test
    void crossThreadArenasPassOnBothThreads() throws Exception {
        try (Arena shared = Arena.ofShared();
                var worker = Executors.newSingleThreadExecutor()) {
            for (Arena arena : new Arena[] {shared, Arena.ofAuto(), Arena.global()}) {
                assertDoesNotThrow(() -> Arenas.requireCrossThread(arena));
                worker.submit(() -> Arenas.requireCrossThread(arena)).get(5, TimeUnit.SECONDS);
                assertTrue(arena.scope().isAlive());
            }
        }
    }

    @Test
    void contextStateDiagnosesBorrowedConfinedMemory() {
        try (Arena arena = Arena.ofConfined()) {
            assertThrows(
                    IllegalArgumentException.class,
                    () ->
                            new RuntimeStateLifecycleTest.ProbeState(
                                    MemoryAllocators.ofArena(arena), false));
            assertTrue(arena.scope().isAlive());
        }
    }
}
