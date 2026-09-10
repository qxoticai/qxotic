package com.qxotic.jinfer;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.memory.MemoryAllocators;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

class ArenasTest {

    @Test
    void confinedArenasFailOnTheirOwnerAndOnASingleWorker() throws Exception {
        try (Arena arena = Arena.ofConfined();
                var worker = Executors.newSingleThreadExecutor()) {
            var owner = assertThrows(AssertionError.class, () -> Arenas.assertCrossThread(arena));
            var other =
                    worker.submit(
                                    () ->
                                            assertThrows(
                                                    AssertionError.class,
                                                    () -> Arenas.assertCrossThread(arena)))
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
                assertDoesNotThrow(() -> Arenas.assertCrossThread(arena));
                worker.submit(() -> Arenas.assertCrossThread(arena)).get(5, TimeUnit.SECONDS);
                assertTrue(arena.scope().isAlive());
            }
        }
    }

    @Test
    void contextStateDiagnosesBorrowedConfinedMemory() {
        try (Arena arena = Arena.ofConfined()) {
            assertThrows(
                    AssertionError.class,
                    () ->
                            new RuntimeStateLifecycleTest.ProbeState(
                                    MemoryAllocators.ofArena(arena), false));
            assertTrue(arena.scope().isAlive());
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void probingIsStrictlyOptIn(boolean enabled) throws Exception {
        Process process =
                new ProcessBuilder(
                                Path.of(System.getProperty("java.home"), "bin", "java").toString(),
                                enabled ? "-ea:com.qxotic.jinfer..." : "-da",
                                "--enable-native-access=ALL-UNNAMED",
                                "-cp",
                                System.getProperty(
                                        "surefire.test.class.path",
                                        System.getProperty("java.class.path")),
                                AssertionProbe.class.getName(),
                                Boolean.toString(enabled))
                        .redirectErrorStream(true)
                        .start();
        try {
            assertTrue(process.waitFor(10, TimeUnit.SECONDS), "assertion probe timed out");
            String output =
                    new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
            assertEquals(0, process.exitValue(), output);
        } finally {
            if (process.isAlive()) process.destroyForcibly();
        }
    }

    public static final class AssertionProbe {
        public static void main(String[] args) {
            boolean enabled = Boolean.parseBoolean(args[0]);
            AtomicInteger allocations = new AtomicInteger();
            try (Arena shared = Arena.ofShared()) {
                Arena counted =
                        new Arena() {
                            @Override
                            public MemorySegment allocate(long size, long alignment) {
                                allocations.incrementAndGet();
                                return shared.allocate(size, alignment);
                            }

                            @Override
                            public MemorySegment.Scope scope() {
                                return shared.scope();
                            }

                            @Override
                            public void close() {
                                shared.close();
                            }
                        };
                Arenas.assertCrossThread(counted);
                Arenas.assertCrossThread(MemoryAllocators.ofArena(counted));
                if (allocations.get() != (enabled ? 2 : 0)) {
                    throw new AssertionError("unexpected probe count: " + allocations.get());
                }
            }
            // Never perform inference here: with assertions disabled this unsupported input
            // must simply pass the diagnostic boundary without being inspected.
            if (!enabled) {
                try (Arena confined = Arena.ofConfined()) {
                    Arenas.assertCrossThread(confined);
                    Arenas.assertCrossThread(MemoryAllocators.ofArena(confined));
                }
            }
        }
    }
}
