package com.qxotic.jinfer;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.memory.MemoryArena;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.ConcurrentModificationException;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;

/** Runtime-state exclusion, quiescent close and arena ownership. */
class RuntimeStateLifecycleTest {

    static final class ProbeState extends ContextState {
        final MemoryView<MemorySegment> buffer;

        ProbeState(MemoryArena<MemorySegment> arena, boolean ownsArena) {
            super(8, 8, arena, ownsArena);
            buffer = Views.allocateF32(memoryArena(), 8);
        }

        Arena jdkArena() {
            return ((PanamaMemoryArena) memoryArena()).arena();
        }

        @Override
        protected void clearHistory() {}
    }

    static ProbeState owned() {
        return new ProbeState(new PanamaMemoryArena(Arena.ofShared()), true);
    }

    @Test
    void sameThreadMayNestButAnotherThreadFailsFast() throws Exception {
        try (ProbeState state = owned();
                ProbeState independent = owned()) {
            state.exclusively(
                    () -> {
                        state.exclusively(() -> {});
                        independent.exclusively(() -> {});
                        AtomicReference<Throwable> failure = new AtomicReference<>();
                        Thread contender =
                                new Thread(
                                        () -> {
                                            try {
                                                state.exclusively(() -> {});
                                            } catch (Throwable e) {
                                                failure.set(e);
                                            }
                                        });
                        contender.start();
                        try {
                            contender.join(10_000);
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                        }
                        assertTrue(failure.get() instanceof ConcurrentModificationException);
                    });
        }
    }

    @Test
    void closeWaitsForTheRunningOperationThenRejectsFurtherUse() throws Exception {
        ProbeState state = owned();
        CountDownLatch entered = new CountDownLatch(1);
        CountDownLatch finish = new CountDownLatch(1);
        Thread worker =
                new Thread(
                        () ->
                                state.exclusively(
                                        () -> {
                                            entered.countDown();
                                            try {
                                                finish.await();
                                            } catch (InterruptedException e) {
                                                Thread.currentThread().interrupt();
                                            }
                                        }));
        worker.start();
        assertTrue(entered.await(10, TimeUnit.SECONDS));

        AtomicReference<Long> closedAt = new AtomicReference<>();
        Thread closer =
                new Thread(
                        () -> {
                            state.close();
                            closedAt.set(System.nanoTime());
                        });
        closer.start();
        Thread.sleep(50);
        assertTrue(closedAt.get() == null, "close returned before the operation finished");
        long releasedAt = System.nanoTime();
        finish.countDown();
        worker.join(10_000);
        closer.join(10_000);

        assertTrue(closedAt.get() >= releasedAt);
        assertThrows(IllegalStateException.class, () -> state.exclusively(() -> {}));
        assertDoesNotThrow(state::close);
    }

    @Test
    void ownedMemoryDiesButBorrowedMemoryDoesNot() {
        ProbeState owned = owned();
        Arena ownedArena = owned.jdkArena();
        owned.close();
        assertFalse(ownedArena.scope().isAlive());

        try (Arena arena = Arena.ofShared()) {
            ProbeState borrowed = new ProbeState(new PanamaMemoryArena(arena), false);
            borrowed.close();
            assertTrue(arena.scope().isAlive());
            assertThrows(IllegalStateException.class, () -> borrowed.exclusively(() -> {}));
        }
    }

    @Test
    void nonCloseableOwnedArenasAreSafe() {
        assertDoesNotThrow(
                () -> new ProbeState(new PanamaMemoryArena(Arena.ofAuto()), true).close());
        assertDoesNotThrow(
                () -> new ProbeState(new PanamaMemoryArena(Arena.global()), true).close());
    }

    @Test
    void runtimeArenaCloseIsIdempotent() {
        Arena arena = Arenas.newCrossThread();
        Arenas.close(arena);
        assertDoesNotThrow(() -> Arenas.close(arena));
    }

    @Test
    void closeFromInsideAnOperationIsRejectedWithoutPoisoningTheState() {
        ProbeState state = owned();
        state.exclusively(() -> assertThrows(IllegalStateException.class, state::close));
        assertDoesNotThrow(() -> state.exclusively(() -> {}));
        state.close();
    }

    @Test
    void cleanupRunsOnceAndItsFailureIsStable() {
        AtomicInteger releases = new AtomicInteger();
        IllegalStateException expected = new IllegalStateException("release failed");
        RuntimeState state =
                new RuntimeState() {
                    @Override
                    protected void releaseResources() {
                        releases.incrementAndGet();
                        throw expected;
                    }
                };

        assertSame(expected, assertThrows(IllegalStateException.class, state::close));
        assertSame(expected, assertThrows(IllegalStateException.class, state::close));
        assertEquals(1, releases.get());
        assertFalse(state.isAlive());
    }
}
