package com.qxotic.jinfer.x.boundary;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.x.PanamaMemoryArena;
import com.qxotic.jota.memory.MemoryArena;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;

/** The public state factories make ownership unambiguous: owned or borrowed, never a flag. */
class ModelArenaMatrixTest {

    record Configuration(int vocabularySize, int contextLength) implements ContextConfiguration {}

    static class ProbeModel
            implements LanguageModel<Configuration, Void, RuntimeStateLifecycleTest.ProbeState> {

        @Override
        public Configuration configuration() {
            return new Configuration(8, 8);
        }

        @Override
        public Void weights() {
            return null;
        }

        @Override
        public RuntimeStateLifecycleTest.ProbeState newState(
                int contextCapacity, int batchCapacity) {
            MemoryArena<MemorySegment> arena = Arenas.newCrossThreadMemoryArena();
            try {
                return create(arena, true);
            } catch (RuntimeException | Error failure) {
                Arenas.close(arena);
                throw failure;
            }
        }

        @Override
        public RuntimeStateLifecycleTest.ProbeState newState(
                int contextCapacity, int batchCapacity, MemoryArena<MemorySegment> arena) {
            return create(arena, false);
        }

        RuntimeStateLifecycleTest.ProbeState create(
                MemoryArena<MemorySegment> arena, boolean ownsArena) {
            return new RuntimeStateLifecycleTest.ProbeState(arena, ownsArena);
        }

        @Override
        public void ingest(RuntimeStateLifecycleTest.ProbeState state, Batch batch) {
            state.exclusively(() -> {});
        }

        @Override
        public com.qxotic.jota.memory.MemoryView<?> logits(
                RuntimeStateLifecycleTest.ProbeState state, int output) {
            return state.exclusively(() -> state.buffer);
        }
    }

    private final ProbeModel model = new ProbeModel();

    @Test
    void ownedFactoryFreesItsPrivateArena() {
        RuntimeStateLifecycleTest.ProbeState state = model.newState(8, 8);
        Arena arena = state.jdkArena();
        state.close();
        assertFalse(arena.scope().isAlive());
    }

    @Test
    void borrowedFactoryNeverClosesTheCallersArena() {
        try (Arena arena = Arena.ofShared()) {
            RuntimeStateLifecycleTest.ProbeState state =
                    model.newState(8, 8, new PanamaMemoryArena(arena));
            state.close();
            assertTrue(arena.scope().isAlive());
        }
    }

    @Test
    void noPublicFactoryExposesAnOwnershipBoolean() {
        assertFalse(
                Arrays.stream(ContextModel.class.getMethods())
                        .filter(method -> method.getName().equals("newState"))
                        .flatMap(method -> Arrays.stream(method.getParameterTypes()))
                        .anyMatch(type -> type == boolean.class));
    }

    @Test
    void ownedFactoryClosesItsArenaWhenConstructionFails() {
        AtomicReference<MemoryArena<MemorySegment>> seen = new AtomicReference<>();
        ProbeModel failing =
                new ProbeModel() {
                    @Override
                    RuntimeStateLifecycleTest.ProbeState create(
                            MemoryArena<MemorySegment> arena, boolean ownsArena) {
                        seen.set(arena);
                        throw new IllegalArgumentException("family constructor failure");
                    }
                };

        assertThrows(IllegalArgumentException.class, () -> failing.newState(8, 8));
        assertFalse(seen.get().isAlive());
    }
}
