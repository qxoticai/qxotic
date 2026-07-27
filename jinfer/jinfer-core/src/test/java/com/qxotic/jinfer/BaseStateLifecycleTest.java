package com.qxotic.jinfer;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.foreign.Arena;
import java.util.ConcurrentModificationException;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;

/**
 * The state lifetime/concurrency laws, pinned: entries fail fast with CME under concurrent use
 * (never queued - queuing would hide the contract violation), close() BLOCKS until the in-flight
 * computation returns (the caller's quiescence certificate), entries after close fail with ISE
 * before any kernel could touch freed memory, double close is a no-op, and an owned state's arena
 * is actually freed by close.
 */
class BaseStateLifecycleTest {

    /** Minimal concrete state: one probe buffer from the arena, plus the inherited lifecycle. */
    static final class ProbeState extends BaseState {
        final F32FloatTensor buf;

        ProbeState(Arena arena) {
            super(arena);
            this.buf = F32FloatTensor.allocate(arena, 8);
        }

        @Override
        public int contextCapacity() {
            return 8;
        }

        @Override
        public int batchCapacity() {
            return 8;
        }

        @Override
        public void reset() {
            resumeAt(0);
        }
    }

    static ProbeState owned() {
        ProbeState s = new ProbeState(Arena.ofShared());
        s.adoptArena();
        return s;
    }

    @Test
    void concurrentEntryIsCme_reentrantForTheHolder_independentStatesDoNotInterfere()
            throws Exception {
        ProbeState a = owned();
        ProbeState b = owned();
        a.enter();
        a.enter(); // a generation holds across many forwards; nested entry is fine
        b.enter(); // a different state is a different pipeline
        Thread t = new Thread(() -> assertThrows(ConcurrentModificationException.class, a::enter));
        t.start();
        t.join();
        a.exit();
        a.exit();
        b.exit();
        // fully released: any thread may enter again
        Thread t2 = new Thread(() -> assertDoesNotThrow(() -> a.enter()));
        t2.start();
        t2.join();
    }

    @Test
    void closeBlocksUntilTheInFlightComputationReturns() throws Exception {
        ProbeState s = owned();
        CountDownLatch entered = new CountDownLatch(1);
        CountDownLatch finish = new CountDownLatch(1);
        Thread worker =
                new Thread(
                        () -> {
                            s.enter();
                            entered.countDown();
                            try {
                                finish.await();
                            } catch (InterruptedException ignored) {
                            } finally {
                                s.exit();
                            }
                        });
        worker.start();
        entered.await();
        AtomicReference<Long> closedAt = new AtomicReference<>();
        Thread closer =
                new Thread(
                        () -> {
                            s.close(); // must block until the worker exits
                            closedAt.set(System.nanoTime());
                        });
        closer.start();
        Thread.sleep(50); // give the closer time to reach the lock
        assertTrue(closedAt.get() == null, "close returned while the computation was in flight");
        long releasedAt = System.nanoTime();
        finish.countDown();
        worker.join();
        closer.join();
        assertTrue(closedAt.get() >= releasedAt, "close must complete after the worker released");
        assertTrue(s.isClosed());
    }

    @Test
    void entryAfterCloseIsIse_doubleCloseIsNoOp_closeActuallyFreesTheOwnedArena() {
        ProbeState s = owned();
        Arena arena = s.arena;
        assertTrue(arena.scope().isAlive());
        s.close();
        assertFalse(arena.scope().isAlive(), "owned arena must be freed by close, not by GC");
        assertThrows(IllegalStateException.class, s::enter);
        assertDoesNotThrow(s::close); // idempotent
    }

    @Test
    void borrowedArenaIsNeverClosedByTheState() {
        try (Arena arena = Arena.ofShared()) {
            ProbeState s = new ProbeState(arena); // borrowed: no adoptArena
            s.close();
            assertTrue(arena.scope().isAlive(), "close must not touch a caller-owned arena");
            assertThrows(IllegalStateException.class, s::enter); // but the state is still dead
        }
    }

    @Test
    void adoptFusesLifetimes_everythingInTheArenaDiesWithTheState() {
        Arena arena = Arena.ofShared();
        F32FloatTensor.allocate(arena, 8); // a weights-like co-tenant, allocated before the state
        ProbeState s = new ProbeState(arena);
        s.adoptArena();
        s.close();
        assertFalse(arena.scope().isAlive(), "adopt: the caller-created arena dies with the state");
    }

    @Test
    void adoptedNonCloseableArenaMakesCloseANoOpOnTheMemory() {
        ProbeState auto = new ProbeState(Arena.ofAuto());
        auto.adoptArena();
        assertDoesNotThrow(auto::close); // ofAuto/global manage themselves; nothing to free eagerly
        assertTrue(auto.isClosed());
        ProbeState global = new ProbeState(Arena.global());
        global.adoptArena();
        assertDoesNotThrow(global::close);
        assertTrue(global.isClosed());
    }

    @Test
    void closeFromWithinTheOwnComputationIsRejected() {
        ProbeState s = owned();
        s.enter();
        try {
            assertThrows(IllegalStateException.class, s::close);
        } finally {
            s.exit();
        }
        assertFalse(s.isClosed(), "the rejected close must not have marked the state dead");
        s.close();
    }
}
