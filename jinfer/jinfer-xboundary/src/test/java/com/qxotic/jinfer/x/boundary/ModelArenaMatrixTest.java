package com.qxotic.jinfer.x.boundary;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.foreign.Arena;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;

/**
 * The arena-ownership matrix through the {@link Model} API (not BaseState directly): every {@code
 * newState} flavor x every arena kind a caller can legally hand over. Owned frees on close;
 * borrowed is never touched; adopt fuses; non-closeable arenas (ofAuto/global) are safe to borrow
 * AND to adopt; a family constructor throwing inside the owned flavor must not leak the fresh
 * internal arena.
 */
class ModelArenaMatrixTest {

    /** Minimal model over the lifecycle-test state: forward is a no-op, weights are nothing. */
    static class ProbeModel implements Model<Config, Void, BaseStateLifecycleTest.ProbeState> {
        @Override
        public Config config() {
            return new Config() {
                @Override
                public int vocabularySize() {
                    return 8;
                }

                @Override
                public int contextLength() {
                    return 8;
                }
            };
        }

        @Override
        public Void weights() {
            return null;
        }

        @Override
        public BaseStateLifecycleTest.ProbeState newState(
                int contextCapacity, int batchCapacity, Arena arena) {
            return new BaseStateLifecycleTest.ProbeState(arena);
        }

        @Override
        public void forward(BaseStateLifecycleTest.ProbeState state, Batch batch) {}
    }

    final ProbeModel model = new ProbeModel();

    @Test
    void ownedStateFreesItsInternalArenaOnClose() {
        BaseStateLifecycleTest.ProbeState s = model.newState(8, 8);
        Arena arena = s.jdkArena();
        assertTrue(arena.scope().isAlive());
        s.close();
        assertFalse(arena.scope().isAlive(), "owned: the internal arena dies with the state");
    }

    @Test
    void borrowedArenaIsNeverTouched() {
        try (Arena arena = Arena.ofShared()) {
            BaseStateLifecycleTest.ProbeState s = model.newState(8, arena);
            s.close();
            assertTrue(
                    arena.scope().isAlive(), "borrowed: close must not touch the caller's arena");
            assertThrows(IllegalStateException.class, s::enter);
        }
    }

    @Test
    void adoptTrueFusesTheCallerArenaIntoTheState() {
        Arena arena = Arena.ofShared();
        arena.allocate(32); // a weights-like co-tenant
        BaseStateLifecycleTest.ProbeState s = model.newState(8, 8, arena, true);
        s.close();
        assertFalse(arena.scope().isAlive(), "adopt: the caller's arena dies with the state");
    }

    @Test
    void adoptFalseIsBorrowed() {
        try (Arena arena = Arena.ofShared()) {
            BaseStateLifecycleTest.ProbeState s = model.newState(8, 8, arena, false);
            s.close();
            assertTrue(arena.scope().isAlive());
        }
    }

    @Test
    void nonCloseableArenasAreSafeBorrowedAndAdopted() {
        for (Arena arena : new Arena[] {Arena.ofAuto(), Arena.global()}) {
            BaseStateLifecycleTest.ProbeState borrowed = model.newState(8, arena);
            assertDoesNotThrow(borrowed::close);
            BaseStateLifecycleTest.ProbeState adopted = model.newState(8, 8, arena, true);
            assertDoesNotThrow(adopted::close); // owning ofAuto/global = nothing to free eagerly
            assertTrue(adopted.isClosed());
        }
    }

    @Test
    void familyConstructorThrowInOwnedFlavorFreesTheFreshArena() {
        AtomicReference<Arena> seen = new AtomicReference<>();
        ProbeModel failing =
                new ProbeModel() {
                    @Override
                    public BaseStateLifecycleTest.ProbeState newState(
                            int contextCapacity, int batchCapacity, Arena arena) {
                        seen.set(arena);
                        throw new IllegalArgumentException("family ctor failure");
                    }
                };
        assertThrows(IllegalArgumentException.class, () -> failing.newState(8, 8));
        assertFalse(
                seen.get().scope().isAlive(),
                "a leaked ofShared arena has no Cleaner: the owned flavor must free it on throw");
    }
}
