package com.qxotic.jam.vector;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicIntegerArray;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;

final class VectorSupportParallelTest {

    @Test
    void configuredParallelismIsApplied() {
        String expected = System.getProperty("jam.vector.threads");
        if (expected == null) expected = System.getProperty("jam.threads");
        assumeTrue(expected != null);
        assertEquals(Integer.parseInt(expected), VectorSupport.PARALLELISM);
    }

    @Test
    void visitsEveryIndexOnceOnTheVectorPool() {
        assumeTrue(VectorSupport.PARALLELISM > 1);
        AtomicIntegerArray visits = new AtomicIntegerArray(VectorSupport.PARALLELISM * 8);
        Set<ForkJoinPool> pools = ConcurrentHashMap.newKeySet();

        VectorSupport.parallelFor(
                0,
                visits.length(),
                i -> {
                    pools.add(ForkJoinTask.getPool());
                    visits.incrementAndGet(i);
                });

        assertEquals(1, pools.size());
        ForkJoinPool pool = pools.iterator().next();
        assertNotSame(ForkJoinPool.commonPool(), pool);
        assertEquals(VectorSupport.PARALLELISM, pool.getParallelism());
        for (int i = 0; i < visits.length(); i++) assertEquals(1, visits.get(i), "index " + i);
    }

    @Test
    void failureWaitsForRunningSibling() throws Exception {
        assumeTrue(VectorSupport.PARALLELISM > 1);
        IllegalStateException marker = new IllegalStateException("marker");
        CountDownLatch siblingStarted = new CountDownLatch(1);
        CountDownLatch releaseSibling = new CountDownLatch(1);
        AtomicReference<Throwable> failure = new AtomicReference<>();

        Thread caller =
                Thread.ofPlatform()
                        .start(
                                () -> {
                                    try {
                                        VectorSupport.parallelFor(
                                                0,
                                                2,
                                                i -> {
                                                    if (i == 0) throw marker;
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

        assertTrue(siblingStarted.await(5, TimeUnit.SECONDS));
        assertTrue(caller.isAlive());
        releaseSibling.countDown();
        caller.join(5_000);
        assertSame(marker, failure.get());
    }
}
