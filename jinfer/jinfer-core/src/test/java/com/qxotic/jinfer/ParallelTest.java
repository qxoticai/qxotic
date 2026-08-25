package com.qxotic.jinfer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicIntegerArray;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;

final class ParallelTest {

    @Test
    void forLoopVisitsEveryIndexExactlyOnceOnTheJinferPool() {
        Thread caller = Thread.currentThread();
        AtomicIntegerArray visits = new AtomicIntegerArray(257);
        Set<ForkJoinPool> pools = ConcurrentHashMap.newKeySet();

        Parallel.forLoop(
                visits.length(),
                i -> {
                    assertNotSame(caller, Thread.currentThread());
                    pools.add(ForkJoinTask.getPool());
                    visits.incrementAndGet(i);
                });

        assertEquals(1, pools.size());
        ForkJoinPool pool = pools.iterator().next();
        assertNotSame(ForkJoinPool.commonPool(), pool);
        assertEquals(RuntimeFlags.COMPUTE_THREADS, pool.getParallelism());
        for (int i = 0; i < visits.length(); i++) {
            assertEquals(1, visits.get(i), "index " + i);
        }
    }

    @Test
    void singleElementRangeRunsInline() {
        Thread caller = Thread.currentThread();

        Parallel.forLoop(
                7,
                8,
                i -> {
                    assertEquals(7, i);
                    assertSame(caller, Thread.currentThread());
                });
    }

    @Test
    void emptyRangesDoNoWork() {
        Parallel.forLoop(4, 4, ignored -> fail("empty range ran"));
        Parallel.forLoop(5, 4, ignored -> fail("reversed range ran"));
    }

    @Test
    void nestedLoopsStayOnTheSamePool() {
        Set<ForkJoinPool> pools = ConcurrentHashMap.newKeySet();
        AtomicIntegerArray visits = new AtomicIntegerArray(64);

        Parallel.forLoop(
                0,
                8,
                row ->
                        Parallel.forLoop(
                                0,
                                8,
                                column -> {
                                    pools.add(ForkJoinTask.getPool());
                                    visits.incrementAndGet(row * 8 + column);
                                }));

        assertEquals(1, pools.size());
        for (int i = 0; i < visits.length(); i++) {
            assertEquals(1, visits.get(i), "cell " + i);
        }
    }

    @Test
    void saturatedNestedLoopsCompleteOnTheBoundedPool() {
        int repetitions = 100;
        int threads = RuntimeFlags.COMPUTE_THREADS;
        int outerCount = threads * 4;
        int innerCount = 256;
        AtomicLong visits = new AtomicLong();

        for (int repetition = 0; repetition < repetitions; repetition++) {
            AtomicInteger entered = new AtomicInteger();
            Parallel.forLoop(
                    outerCount,
                    ignored -> {
                        if (entered.getAndIncrement() < threads) {
                            while (entered.get() < threads) Thread.onSpinWait();
                        }
                        Parallel.forLoop(innerCount, inner -> visits.incrementAndGet());
                    });
        }

        assertEquals((long) repetitions * outerCount * innerCount, visits.get());
    }

    @Test
    void decodeStepFailuresPropagateTheOriginalInstance() throws Exception {
        assumeTrue(RuntimeFlags.DECODE_SPIN);
        IllegalStateException marker = new IllegalStateException("marker");

        // spin path: sole submitter, the step throws raw
        try {
            Parallel.runDecodeStep(
                    () -> {
                        throw marker;
                    });
            fail("expected the marker");
        } catch (Throwable t) {
            assertSame(marker, t);
        }

        // contended fallback: a second submitter is routed to the decode ForkJoinPool; an external
        // join() would surface a reflective copy (ForkJoinTask.getException) - pin the raw one
        CountDownLatch inSpin = new CountDownLatch(1);
        CountDownLatch releaseSpin = new CountDownLatch(1);
        AtomicReference<Throwable> caught = new AtomicReference<>();
        Thread holder =
                Thread.ofPlatform()
                        .start(
                                () ->
                                        Parallel.runDecodeStep(
                                                () -> {
                                                    inSpin.countDown();
                                                    try {
                                                        releaseSpin.await();
                                                    } catch (InterruptedException interrupted) {
                                                        Thread.currentThread().interrupt();
                                                        throw new AssertionError(interrupted);
                                                    }
                                                    return null;
                                                }));
        Thread contender =
                Thread.ofPlatform()
                        .start(
                                () -> {
                                    try {
                                        assertTrue(inSpin.await(5, TimeUnit.SECONDS));
                                        Parallel.runDecodeStep(
                                                () -> {
                                                    throw marker;
                                                });
                                    } catch (InterruptedException interrupted) {
                                        Thread.currentThread().interrupt();
                                        caught.set(interrupted);
                                    } catch (Throwable thrown) {
                                        caught.set(thrown);
                                    } finally {
                                        releaseSpin.countDown();
                                    }
                                });
        holder.join(5_000);
        contender.join(5_000);
        assertSame(marker, caught.get());
    }

    @Test
    void failureStillWaitsForRunningSiblings() throws Exception {
        assumeTrue(RuntimeFlags.COMPUTE_THREADS > 1);
        CountDownLatch siblingStarted = new CountDownLatch(1);
        CountDownLatch releaseSibling = new CountDownLatch(1);
        AtomicReference<Throwable> failure = new AtomicReference<>();

        Thread caller =
                Thread.ofPlatform()
                        .start(
                                () -> {
                                    try {
                                        Parallel.forLoop(
                                                0,
                                                2,
                                                i -> {
                                                    if (i == 0) {
                                                        throw new IllegalStateException("failed");
                                                    }
                                                    siblingStarted.countDown();
                                                    try {
                                                        releaseSibling.await();
                                                    } catch (InterruptedException interrupted) {
                                                        Thread.currentThread().interrupt();
                                                        throw new AssertionError(interrupted);
                                                    }
                                                });
                                    } catch (Throwable thrown) {
                                        failure.set(thrown);
                                    }
                                });

        try {
            assertTrue(siblingStarted.await(5, TimeUnit.SECONDS));
            assertTrue(caller.isAlive(), "forLoop returned while an action was still running");
        } finally {
            releaseSibling.countDown();
            caller.join(5_000);
        }
        assertFalse(caller.isAlive());
        assertNotNull(failure.get());
    }

    @Test
    void sameFailureFromSiblingTasksIsNotSelfSuppressed() {
        assumeTrue(RuntimeFlags.COMPUTE_THREADS > 1);
        IllegalStateException marker = new IllegalStateException("marker");

        Throwable thrown =
                assertThrows(
                        IllegalStateException.class,
                        () ->
                                Parallel.forLoop(
                                        0,
                                        2,
                                        ignored -> {
                                            throw marker;
                                        }));

        assertSame(marker, thrown);
    }
}
