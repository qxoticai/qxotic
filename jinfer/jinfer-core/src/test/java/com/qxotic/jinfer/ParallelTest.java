package com.qxotic.jinfer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

import java.lang.management.ManagementFactory;
import java.lang.management.ThreadMXBean;
import java.lang.ref.WeakReference;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicIntegerArray;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.atomic.AtomicReferenceArray;
import java.util.concurrent.locks.LockSupport;
import java.util.function.IntConsumer;
import java.util.stream.IntStream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/** The pool's contract, corner by corner: see {@link ParallelFuzzTest} for the randomized runs. */
class ParallelTest {

    // ---- construction and the shared instance ----

    @Test
    void widthMustBePositive() {
        assertThrows(IllegalArgumentException.class, () -> Parallel.of(0));
        assertThrows(IllegalArgumentException.class, () -> Parallel.of(-3));
        assertThrows(IllegalArgumentException.class, () -> Parallel.of(Integer.MIN_VALUE));
    }

    @Test
    void sharedIsOneInstanceSizedByTheFlag() {
        assertSame(Parallel.shared(), Parallel.shared());
        assertEquals(RuntimeFlags.THREADS, Parallel.threads());
        assertEquals(Parallel.shared().width(), Parallel.threads());
        AtomicInteger visits = new AtomicInteger();
        Parallel.forLoop(5, i -> visits.incrementAndGet());
        Parallel.forLoop(2, 5, i -> visits.incrementAndGet());
        Parallel.forLoop(5, (i, slot) -> visits.incrementAndGet());
        assertEquals(13, visits.get());
    }

    @Test
    void widthOneNeverStartsAThreadAndRunsOnTheCaller() {
        Set<String> before = threadNames();
        try (Parallel one = Parallel.of(1)) {
            assertEquals(1, one.width());
            Thread me = Thread.currentThread();
            AtomicInteger slots = new AtomicInteger();
            one.loop(
                    0,
                    1000,
                    (i, slot) -> {
                        assertSame(me, Thread.currentThread());
                        slots.addAndGet(slot);
                    });
            assertEquals(0, slots.get(), "slot 0 everywhere");
        }
        assertEquals(before, threadNames());
    }

    // ---- ranges ----

    @Test
    void emptyAndReversedRangesDoNothing() {
        try (Parallel pool = Parallel.of(4)) {
            pool.loop(0, ignored -> fail("count 0 ran"));
            pool.loop(4, 4, ignored -> fail("empty range ran"));
            pool.loop(5, 4, ignored -> fail("reversed range ran"));
            pool.loop(-7, ignored -> fail("negative count ran"));
            pool.loop(Integer.MAX_VALUE, Integer.MIN_VALUE, ignored -> fail("wrapped range ran"));
        }
    }

    @Test
    void singleElementRangeRunsInlineWithSlotZero() {
        try (Parallel pool = Parallel.of(4)) {
            Thread[] seen = new Thread[1];
            int[] seenSlot = {-1};
            pool.loop(
                    3,
                    4,
                    (i, slot) -> {
                        assertEquals(3, i);
                        seen[0] = Thread.currentThread();
                        seenSlot[0] = slot;
                    });
            assertSame(Thread.currentThread(), seen[0]);
            assertEquals(0, seenSlot[0]);
        }
    }

    @ParameterizedTest
    @ValueSource(ints = {2, 3, 5, 7, 16, 33})
    void everyIndexExactlyOnceAtAnyWidth(int width) {
        int[][] ranges = {
            {0, 2},
            {0, 3},
            {0, 17},
            {-5, 5},
            {100, 101 + width},
            {0, 4 * width - 1},
            {0, 4 * width},
            {0, 4 * width + 1},
            {0, 9973},
            {-1000, 1000},
            {0, 1 << 17}
        };
        try (Parallel pool = Parallel.of(width)) {
            for (int[] r : ranges) {
                int start = r[0], end = r[1];
                AtomicIntegerArray visits = new AtomicIntegerArray(end - start);
                Set<Thread> threads = ConcurrentHashMap.newKeySet();
                pool.loop(
                        start,
                        end,
                        i -> {
                            visits.incrementAndGet(i - start);
                            threads.add(Thread.currentThread());
                        });
                for (int i = 0; i < visits.length(); i++)
                    assertEquals(1, visits.get(i), "range " + start + ".." + end + " index " + i);
                assertTrue(threads.size() <= width, "at most the budget");
            }
        }
    }

    @Test
    void largeCountsUseTheWholePool() {
        try (Parallel pool = Parallel.of(8)) {
            Set<Thread> threads = ConcurrentHashMap.newKeySet();
            AtomicLong sum = new AtomicLong();
            pool.loop(
                    1 << 16,
                    i -> {
                        threads.add(Thread.currentThread());
                        sum.addAndGet(i);
                        spin(2_000);
                    });
            assertEquals((1L << 16) * ((1L << 16) - 1) / 2, sum.get());
            assertEquals(8, threads.size(), "65k 2-microsecond items spread over all 8");
            assertTrue(threads.contains(Thread.currentThread()), "the caller participates");
        }
    }

    // ---- slots ----

    @Test
    void slotsAreInRangeUniqueAmongLiveTasksAndStableAcrossRegions() {
        int width = 6;
        try (Parallel pool = Parallel.of(width)) {
            AtomicIntegerArray busy = new AtomicIntegerArray(width);
            ConcurrentHashMap<Thread, Integer> slotOf = new ConcurrentHashMap<>();
            Thread caller = Thread.currentThread();
            for (int round = 0; round < 200; round++) {
                pool.loop(
                        0,
                        width * 8,
                        (i, slot) -> {
                            assertTrue(slot >= 0 && slot < width, "slot " + slot);
                            assertTrue(busy.compareAndSet(slot, 0, 1), "slot " + slot + " busy");
                            Integer previous = slotOf.put(Thread.currentThread(), slot);
                            if (previous != null) assertEquals(previous, slot, "stable slot");
                            if (Thread.currentThread() == caller) assertEquals(0, slot);
                            spin(20_000);
                            busy.set(slot, 0);
                        });
            }
            assertEquals(0, slotOf.get(caller), "the caller is slot 0");
        }
    }

    @Test
    void tinyRegionsStillGetDistinctSlots() {
        try (Parallel pool = Parallel.of(8)) {
            for (int round = 0; round < 100; round++) {
                Set<Integer> slots = ConcurrentHashMap.newKeySet();
                CountDownLatch both = new CountDownLatch(2);
                pool.loop(
                        2,
                        (i, slot) -> {
                            slots.add(slot);
                            both.countDown();
                            await(both);
                        });
                assertEquals(2, slots.size(), "two live tasks, two slots");
            }
        }
    }

    // ---- nesting ----

    @Test
    void nestedLoopsRunInlineOnTheSameSlot() {
        try (Parallel pool = Parallel.of(4)) {
            AtomicIntegerArray visits = new AtomicIntegerArray(64);
            pool.loop(
                    0,
                    8,
                    (row, outerSlot) -> {
                        Thread outer = Thread.currentThread();
                        pool.loop(
                                0,
                                8,
                                (column, innerSlot) -> {
                                    assertSame(outer, Thread.currentThread());
                                    assertEquals(outerSlot, innerSlot);
                                    visits.incrementAndGet(row * 8 + column);
                                });
                    });
            for (int i = 0; i < visits.length(); i++) assertEquals(1, visits.get(i), "cell " + i);
        }
    }

    @Test
    void deepRecursionInsideARegionIsInline() {
        try (Parallel pool = Parallel.of(4)) {
            AtomicInteger leaves = new AtomicInteger();
            pool.loop(4, i -> recurse(pool, 60, leaves));
            assertEquals(4, leaves.get());
        }
    }

    private static void recurse(Parallel pool, int depth, AtomicInteger leaves) {
        if (depth == 0) {
            leaves.incrementAndGet();
            return;
        }
        pool.loop(
                2,
                i -> {
                    if (i == 0) recurse(pool, depth - 1, leaves);
                });
    }

    @Test
    void saturatedNestedLoopsComplete() {
        try (Parallel pool = Parallel.of(6)) {
            int outerCount = 24, innerCount = 256;
            AtomicLong visits = new AtomicLong();
            for (int repetition = 0; repetition < 100; repetition++) {
                pool.loop(outerCount, i -> pool.loop(innerCount, j -> visits.incrementAndGet()));
            }
            assertEquals(100L * outerCount * innerCount, visits.get());
        }
    }

    @Test
    void aLoopOnAnotherPoolFromInsideARegionIsARealRegionThere() {
        try (Parallel a = Parallel.of(3);
                Parallel b = Parallel.of(4)) {
            Set<Thread> insideB = ConcurrentHashMap.newKeySet();
            AtomicInteger visits = new AtomicInteger();
            a.loop(
                    0,
                    2000,
                    i ->
                            b.loop(
                                    0,
                                    64,
                                    j -> {
                                        insideB.add(Thread.currentThread());
                                        visits.incrementAndGet();
                                    }));
            assertEquals(2000 * 64, visits.get());
            assertTrue(insideB.size() > 1, "b's workers ran b's regions");
        }
    }

    // ---- exceptions ----

    @Test
    void theOriginalExceptionPropagatesAndNoIndexRunsTwice() {
        for (int width : new int[] {2, 3, 5, 16}) {
            try (Parallel pool = Parallel.of(width)) {
                AtomicIntegerArray visits = new AtomicIntegerArray(64);
                RuntimeException boom = new IllegalStateException("boom");
                RuntimeException thrown =
                        assertThrows(
                                IllegalStateException.class,
                                () ->
                                        pool.loop(
                                                0,
                                                64,
                                                i -> {
                                                    visits.incrementAndGet(i);
                                                    if (i == 7) throw boom;
                                                }));
                assertSame(boom, thrown, "width " + width);
                assertEquals(1, visits.get(7));
                for (int i = 0; i < 64; i++) assertTrue(visits.get(i) <= 1, "index " + i);
            }
        }
    }

    @Test
    void errorsPropagateAsThemselves() {
        try (Parallel pool = Parallel.of(4)) {
            Error error = new AssertionError("fatal");
            Error thrown =
                    assertThrows(
                            AssertionError.class,
                            () ->
                                    pool.loop(
                                            0,
                                            100,
                                            i -> {
                                                if (i == 50) throw error;
                                            }));
            assertSame(error, thrown);
        }
    }

    @Test
    void checkedThrowablesAreWrapped() {
        try (Parallel pool = Parallel.of(4)) {
            Exception checked = new java.io.IOException("io");
            RuntimeException thrown =
                    assertThrows(
                            RuntimeException.class,
                            () ->
                                    pool.loop(
                                            0,
                                            100,
                                            i -> {
                                                if (i == 3) sneaky(checked);
                                            }));
            assertSame(checked, thrown.getCause());
        }
    }

    @Test
    void manyFailuresSurfaceOneOfThemAndStopEarly() {
        try (Parallel pool = Parallel.of(5)) {
            Set<Throwable> thrown = ConcurrentHashMap.newKeySet();
            AtomicInteger done = new AtomicInteger();
            RuntimeException got =
                    assertThrows(
                            RuntimeException.class,
                            () ->
                                    pool.loop(
                                            0,
                                            1_000_000,
                                            i -> {
                                                if (i % 10 == 0) {
                                                    RuntimeException e =
                                                            new RuntimeException("" + i);
                                                    thrown.add(e);
                                                    throw e;
                                                }
                                                done.incrementAndGet();
                                            }));
            assertTrue(thrown.contains(got), "one of the thrown instances");
            assertTrue(done.get() < 900_000, "fail fast: " + done.get() + " ran");
        }
    }

    @Test
    void failuresOnTheCallerAndOnWorkersBothPropagate() {
        try (Parallel pool = Parallel.of(4)) {
            for (int failing = 0; failing < 4; failing++) {
                int failSlot = failing;
                AtomicBoolean hit = new AtomicBoolean();
                // A slot only throws once it runs a task; on a busy 4-core runner the others can
                // finish all 64 before a worker wakes, so loop until the slot has taken part.
                RuntimeException thrown = null;
                for (int attempt = 0; attempt < 100 && thrown == null; attempt++) {
                    try {
                        pool.loop(
                                0,
                                4 * 16,
                                (i, slot) -> {
                                    spin(20_000);
                                    if (slot == failSlot && hit.compareAndSet(false, true))
                                        throw new RuntimeException("slot " + slot);
                                });
                    } catch (RuntimeException e) {
                        thrown = e;
                    }
                }
                assertNotNull(thrown, "slot " + failSlot + " never ran a task");
                assertEquals("slot " + failSlot, thrown.getMessage());
            }
        }
    }

    @Test
    void thePoolIsUsableAfterAFailure() {
        try (Parallel pool = Parallel.of(4)) {
            assertThrows(
                    RuntimeException.class,
                    () ->
                            pool.loop(
                                    0,
                                    100,
                                    i -> {
                                        throw new RuntimeException();
                                    }));
            AtomicInteger visits = new AtomicInteger();
            pool.loop(0, 1000, i -> visits.incrementAndGet());
            assertEquals(1000, visits.get());
        }
    }

    @Test
    void aFailureInANestedInlineLoopPropagatesThroughTheOuter() {
        try (Parallel pool = Parallel.of(4)) {
            RuntimeException boom = new RuntimeException("inner");
            AtomicInteger outerDone = new AtomicInteger();
            RuntimeException thrown =
                    assertThrows(
                            RuntimeException.class,
                            () ->
                                    pool.loop(
                                            0,
                                            16,
                                            i -> {
                                                pool.loop(
                                                        0,
                                                        8,
                                                        j -> {
                                                            if (i == 5 && j == 3) throw boom;
                                                        });
                                                outerDone.incrementAndGet();
                                            }));
            assertSame(boom, thrown);
            assertTrue(outerDone.get() <= 15, "the failing outer iteration never completes");
        }
    }

    @Test
    void inlineLoopsStopAtTheThrowLikeAPlainLoop() {
        try (Parallel one = Parallel.of(1)) {
            AtomicInteger done = new AtomicInteger();
            RuntimeException first = new RuntimeException("first");
            RuntimeException thrown =
                    assertThrows(
                            RuntimeException.class,
                            () ->
                                    one.loop(
                                            0,
                                            10,
                                            i -> {
                                                done.incrementAndGet();
                                                if (i == 2) throw first;
                                            }));
            assertSame(first, thrown);
            assertEquals(3, done.get());
        }
    }

    // ---- concurrent submitters ----

    @Test
    void concurrentSubmittersAreSerializedAndAllComplete() throws Exception {
        try (Parallel pool = Parallel.of(4)) {
            int submitters = 6, loops = 200, count = 500;
            AtomicInteger owner = new AtomicInteger(-1); // the region whose bodies are in flight
            AtomicInteger inFlight = new AtomicInteger();
            AtomicBoolean overlapped = new AtomicBoolean();
            AtomicIntegerArray totals = new AtomicIntegerArray(submitters);
            List<Thread> threads = new ArrayList<>();
            AtomicReference<Throwable> failure = new AtomicReference<>();
            for (int s = 0; s < submitters; s++) {
                int id = s;
                Thread t =
                        new Thread(
                                () -> {
                                    try {
                                        for (int l = 0; l < loops; l++) {
                                            int region = id * loops + l;
                                            pool.loop(
                                                    0,
                                                    count,
                                                    i -> {
                                                        inFlight.incrementAndGet();
                                                        int current = owner.get();
                                                        if (current == -1)
                                                            owner.compareAndSet(-1, region);
                                                        else if (current != region)
                                                            overlapped.set(true);
                                                        totals.incrementAndGet(id);
                                                        spin(2_000);
                                                        if (inFlight.decrementAndGet() == 0)
                                                            owner.set(-1);
                                                    });
                                        }
                                    } catch (Throwable e) {
                                        failure.set(e);
                                    }
                                },
                                "submitter-" + s);
                threads.add(t);
                t.start();
            }
            for (Thread t : threads) t.join(TimeUnit.MINUTES.toMillis(2));
            for (Thread t : threads) assertFalse(t.isAlive(), "submitter hung");
            assertTrue(failure.get() == null, String.valueOf(failure.get()));
            for (int s = 0; s < submitters; s++) assertEquals(loops * count, totals.get(s));
            assertFalse(overlapped.get(), "two regions of one pool ran at once");
        }
    }

    @Test
    void twoPoolsRunRegionsAtTheSameTime() throws Exception {
        try (Parallel a = Parallel.of(3);
                Parallel b = Parallel.of(3)) {
            CountDownLatch bothInside = new CountDownLatch(2);
            AtomicBoolean met = new AtomicBoolean();
            Thread other =
                    new Thread(
                            () ->
                                    b.loop(
                                            0,
                                            2,
                                            i -> {
                                                bothInside.countDown();
                                                if (await(bothInside, 5)) met.set(true);
                                            }));
            other.start();
            a.loop(
                    0,
                    2,
                    i -> {
                        bothInside.countDown();
                        if (await(bothInside, 5)) met.set(true);
                    });
            other.join(10_000);
            assertTrue(met.get(), "independent pools do not serialize each other");
        }
    }

    // ---- close ----

    @Test
    void closeStopsTheWorkersAndLaterLoopsRunInline() throws Exception {
        Parallel pool = Parallel.of(4);
        pool.loop(0, 100, i -> {});
        Set<String> workers = workerNames(pool);
        assertEquals(3, workers.size(), "width - 1 workers, named jinfer-N");
        pool.close();
        pool.close(); // idempotent
        waitUntil(() -> threadNames().stream().noneMatch(workers::contains), 5_000);
        Thread[] after = new Thread[1];
        AtomicInteger visits = new AtomicInteger();
        pool.loop(
                0,
                100,
                (i, slot) -> {
                    after[0] = Thread.currentThread();
                    assertEquals(0, slot);
                    visits.incrementAndGet();
                });
        assertSame(Thread.currentThread(), after[0]);
        assertEquals(100, visits.get());
    }

    @Test
    void closeBeforeAnyLoopNeverStartsWorkers() {
        Set<String> before = threadNames();
        Parallel pool = Parallel.of(8);
        pool.close();
        AtomicInteger visits = new AtomicInteger();
        pool.loop(0, 1000, i -> visits.incrementAndGet());
        assertEquals(1000, visits.get());
        assertEquals(before, threadNames());
    }

    @Test
    void closingDuringARegionLetsItComplete() throws Exception {
        Parallel pool = Parallel.of(4);
        CountDownLatch started = new CountDownLatch(1);
        AtomicInteger visits = new AtomicInteger();
        Thread submitter =
                new Thread(
                        () ->
                                pool.loop(
                                        0,
                                        400,
                                        i -> {
                                            started.countDown();
                                            spin(200_000);
                                            visits.incrementAndGet();
                                        }));
        submitter.start();
        assertTrue(started.await(5, TimeUnit.SECONDS));
        pool.close();
        submitter.join(30_000);
        assertFalse(submitter.isAlive(), "region hung after close");
        assertEquals(400, visits.get());
    }

    // ---- idle, wake-up and interrupts ----

    @Test
    void parkedWorkersWakeForEveryRegion() throws Exception {
        try (Parallel pool = Parallel.of(4)) {
            pool.loop(0, 100, i -> {});
            Set<String> workers = workerNames(pool);
            for (int round = 0; round < 30; round++) {
                waitUntil(() -> allParked(workers), 5_000);
                Set<Thread> seen = ConcurrentHashMap.newKeySet();
                // 64 items of 0.5 ms in chunks of 4: a worker that wakes within 2 ms still gets one
                pool.loop(
                        0,
                        64,
                        i -> {
                            seen.add(Thread.currentThread());
                            spin(500_000);
                        });
                assertEquals(4, seen.size(), "every parked worker woke (round " + round + ")");
            }
        }
    }

    @Test
    void anInterruptedSubmitterStillCompletesAndKeepsItsFlag() {
        try (Parallel pool = Parallel.of(4)) {
            Thread.currentThread().interrupt();
            try {
                AtomicInteger visits = new AtomicInteger();
                pool.loop(0, 10_000, i -> visits.incrementAndGet());
                assertEquals(10_000, visits.get());
                assertTrue(Thread.currentThread().isInterrupted(), "flag preserved");
            } finally {
                Thread.interrupted();
            }
        }
    }

    @Test
    void interruptedWorkersKeepWorking() throws Exception {
        try (Parallel pool = Parallel.of(4)) {
            pool.loop(0, 100, i -> {});
            Set<String> workers = workerNames(pool);
            for (Thread t : Thread.getAllStackTraces().keySet())
                if (workers.contains(t.getName())) t.interrupt();
            for (int round = 0; round < 20; round++) {
                if (round == 0) waitUntil(() -> allParked(workers), 5_000);
                Set<Thread> seen = ConcurrentHashMap.newKeySet();
                pool.loop(
                        0,
                        64,
                        i -> {
                            seen.add(Thread.currentThread());
                            spin(50_000);
                        });
                assertEquals(4, seen.size());
            }
        }
    }

    // ---- visibility and balance ----

    @Test
    void workerWritesAreVisibleToTheSubmitterAfterTheLoop() {
        try (Parallel pool = Parallel.of(8)) {
            int n = 100_000;
            int[] out = new int[n];
            for (int round = 0; round < 20; round++) {
                int r = round;
                pool.loop(0, n, i -> out[i] = i ^ r);
                for (int i = 0; i < n; i++) if (out[i] != (i ^ r)) fail("stale read at " + i);
            }
        }
    }

    @Test
    void oneSlowItemDoesNotHoldTheRegion() {
        try (Parallel pool = Parallel.of(4)) {
            pool.loop(0, 1000, i -> {}); // warm the workers
            long slowNanos = 30_000_000L;
            long t0 = System.nanoTime();
            pool.loop(
                    0,
                    4 * 4 * 8,
                    i -> {
                        if (i == 0) spin(slowNanos);
                        else spin(200_000);
                    });
            long took = System.nanoTime() - t0;
            // 127 items of 0.2 ms over 3 other participants = ~8.5 ms; the slow item is 30 ms.
            assertTrue(took < 3 * slowNanos, "took " + took / 1e6 + " ms: chunks did not balance");
        }
    }

    @Test
    void emptyRegionsAreCheap() {
        try (Parallel pool = Parallel.of(4)) {
            for (int i = 0; i < 5_000; i++) pool.loop(0, 16, j -> {});
            long t0 = System.nanoTime();
            int regions = 50_000;
            for (int i = 0; i < regions; i++) pool.loop(0, 16, j -> {});
            long perRegion = (System.nanoTime() - t0) / regions;
            assertTrue(
                    perRegion < 20_000,
                    "per region " + perRegion + " ns (measured ~1 us at width 4)");
        }
    }

    // ---- round two: limits, errors, lifecycle, odd callers ----

    @Test
    void rangesAtTheIntegerLimitsDoNotWrap() {
        try (Parallel pool = Parallel.of(4)) {
            int[][] ranges = {
                {Integer.MAX_VALUE - 100, Integer.MAX_VALUE},
                {Integer.MIN_VALUE, Integer.MIN_VALUE + 100},
                {Integer.MAX_VALUE - 1, Integer.MAX_VALUE},
                {-50, 50}
            };
            for (int[] r : ranges) {
                int start = r[0], end = r[1];
                AtomicIntegerArray visits = new AtomicIntegerArray(end - start);
                pool.loop(
                        start,
                        end,
                        i -> {
                            if (i < start || i >= end)
                                fail("index " + i + " outside " + start + ".." + end);
                            visits.incrementAndGet(i - start);
                        });
                for (int i = 0; i < visits.length(); i++)
                    assertEquals(1, visits.get(i), "" + (start + i));
            }
        }
    }

    @Test
    void aHugeSpanIsCountedInLong() {
        // Integer.MIN_VALUE..Integer.MAX_VALUE has 2^32-1 elements: too many to run, so only the
        // chunking is exercised, by a body that ends the region early through an Error.
        try (Parallel pool = Parallel.of(2)) {
            AtomicInteger seen = new AtomicInteger();
            assertThrows(
                    StopEarly.class,
                    () ->
                            pool.loop(
                                    Integer.MIN_VALUE,
                                    Integer.MAX_VALUE,
                                    i -> {
                                        if (seen.incrementAndGet() > 1_000_000)
                                            throw new StopEarly();
                                    }));
        }
    }

    private static final class StopEarly extends Error {}

    @Test
    void aStackOverflowInAWorkerPropagatesAndThePoolSurvives() {
        try (Parallel pool = Parallel.of(4)) {
            pool.loop(0, 1000, i -> spin(1_000)); // warm: the workers are spinning, not parked
            for (int round = 0; round < 3; round++) {
                AtomicInteger onWorkers = new AtomicInteger();
                assertThrows(
                        StackOverflowError.class,
                        () ->
                                pool.loop(
                                        0,
                                        4 * 4 * 8,
                                        (i, slot) -> {
                                            spin(200_000);
                                            if (slot != 0) {
                                                onWorkers.incrementAndGet();
                                                overflow(0);
                                            }
                                        }));
                assertTrue(onWorkers.get() > 0, "a worker took part");
                AtomicInteger visits = new AtomicInteger();
                pool.loop(0, 1000, i -> visits.incrementAndGet());
                assertEquals(1000, visits.get());
            }
        }
    }

    private static int overflow(int depth) {
        return overflow(depth + 1) + 1;
    }

    @Test
    void closeFromInsideARegionLetsTheRegionFinish() throws Exception {
        Parallel pool = Parallel.of(4);
        AtomicInteger visits = new AtomicInteger();
        pool.loop(
                0,
                256,
                i -> {
                    if (i == 100) pool.close();
                    spin(50_000);
                    visits.incrementAndGet();
                });
        assertEquals(256, visits.get());
        Set<Thread> after = ConcurrentHashMap.newKeySet();
        pool.loop(0, 64, i -> after.add(Thread.currentThread()));
        assertEquals(Set.of(Thread.currentThread()), after, "closed: inline from now on");
    }

    @Test
    void concurrentSubmittersOnAClosedPoolRunInline() throws Exception {
        Parallel pool = Parallel.of(4);
        pool.close();
        AtomicInteger visits = new AtomicInteger();
        List<Thread> threads = new ArrayList<>();
        for (int s = 0; s < 6; s++) {
            Thread t =
                    new Thread(
                            () ->
                                    pool.loop(
                                            0,
                                            1000,
                                            (i, slot) -> {
                                                assertEquals(0, slot);
                                                visits.incrementAndGet();
                                            }));
            threads.add(t);
            t.start();
        }
        for (Thread t : threads) t.join(30_000);
        assertEquals(6000, visits.get());
    }

    @Test
    void aWidePoolWithTinyWork() {
        try (Parallel pool = Parallel.of(64)) {
            assertEquals(64, pool.width());
            AtomicIntegerArray visits = new AtomicIntegerArray(3);
            Set<Integer> slots = ConcurrentHashMap.newKeySet();
            for (int round = 0; round < 50; round++) {
                pool.loop(
                        3,
                        (i, slot) -> {
                            visits.incrementAndGet(i);
                            slots.add(slot);
                        });
            }
            for (int i = 0; i < 3; i++) assertEquals(50, visits.get(i));
            assertTrue(slots.stream().allMatch(s -> s >= 0 && s < 64));
        }
    }

    @Test
    void aVirtualThreadCanSubmit() throws Exception {
        try (Parallel pool = Parallel.of(4)) {
            AtomicInteger visits = new AtomicInteger();
            AtomicReference<Throwable> failure = new AtomicReference<>();
            Thread vt =
                    Thread.ofVirtual()
                            .start(
                                    () -> {
                                        try {
                                            for (int round = 0; round < 20; round++)
                                                pool.loop(0, 5000, i -> visits.incrementAndGet());
                                        } catch (Throwable t) {
                                            failure.set(t);
                                        }
                                    });
            vt.join(30_000);
            assertFalse(vt.isAlive());
            assertTrue(failure.get() == null, String.valueOf(failure.get()));
            assertEquals(20 * 5000, visits.get());
        }
    }

    @Test
    void theSharedPoolInsideAnOwnRegionIsARealRegion() {
        try (Parallel own = Parallel.of(3)) {
            AtomicInteger visits = new AtomicInteger();
            Set<Thread> sharedThreads = ConcurrentHashMap.newKeySet();
            own.loop(
                    0,
                    200,
                    i ->
                            Parallel.forLoop(
                                    0,
                                    64,
                                    j -> {
                                        sharedThreads.add(Thread.currentThread());
                                        visits.incrementAndGet();
                                    }));
            assertEquals(200 * 64, visits.get());
            assertTrue(sharedThreads.size() <= Parallel.threads() + 3);
        }
    }

    @Test
    void anotherPoolsWorkerSubmittingHereGetsARegion() {
        try (Parallel a = Parallel.of(4);
                Parallel b = Parallel.of(4)) {
            Set<Thread> aThreads = ConcurrentHashMap.newKeySet();
            AtomicInteger visits = new AtomicInteger();
            b.loop(
                    0,
                    4 * 8,
                    (i, bSlot) -> {
                        if (bSlot == 0) return; // only b's WORKERS submit to a
                        a.loop(
                                0,
                                256,
                                (j, aSlot) -> {
                                    aThreads.add(Thread.currentThread());
                                    visits.incrementAndGet();
                                    spin(5_000);
                                });
                    });
            assertTrue(visits.get() > 0);
            assertTrue(visits.get() % 256 == 0);
            assertTrue(aThreads.size() <= 4 + 3, "a's workers plus b's submitting workers");
        }
    }

    @Test
    void sleepingBodiesStillComplete() {
        try (Parallel pool = Parallel.of(4)) {
            AtomicInteger visits = new AtomicInteger();
            pool.loop(
                    0,
                    32,
                    i -> {
                        LockSupport.parkNanos(2_000_000);
                        visits.incrementAndGet();
                    });
            assertEquals(32, visits.get());
        }
    }

    @Test
    void aMillionTinyRegions() {
        try (Parallel pool = Parallel.of(4)) {
            AtomicLong visits = new AtomicLong();
            long t0 = System.nanoTime();
            for (int r = 0; r < 300_000; r++) pool.loop(0, 2, i -> visits.incrementAndGet());
            long perRegion = (System.nanoTime() - t0) / 300_000;
            assertEquals(600_000, visits.get());
            assertTrue(
                    perRegion < 20_000,
                    "per region " + perRegion + " ns (measured ~1 us at width 4)");
        }
    }

    @Test
    void runClaimsOneJobAtATimeAndStillVisitsEveryJobOnce() {
        try (Parallel pool = Parallel.of(4)) {
            int n = 256;
            AtomicIntegerArray visits = new AtomicIntegerArray(n);
            Set<Integer> slots = ConcurrentHashMap.newKeySet();
            pool.run(
                    n,
                    (i, slot) -> {
                        visits.incrementAndGet(i);
                        slots.add(slot);
                        spin(50_000); // 256 x 50 us: with one index per claim every slot is busy
                    });
            for (int i = 0; i < n; i++) assertEquals(1, visits.get(i));
            assertEquals(4, slots.size(), "every participant claimed items");
            AtomicInteger visited = new AtomicInteger();
            Parallel.shared().run(10, (i, slot) -> visited.incrementAndGet());
            assertEquals(10, visited.get());
        }
    }

    @Test
    void loopClaimsContiguousHalfBands() {
        try (Parallel pool = Parallel.of(4)) {
            int n = 4096; // 2 x 4 jobs of 512 contiguous indices
            int[] slotOf = new int[n];
            pool.loop(0, n, (i, slot) -> slotOf[i] = slot);
            for (int c = 0; c < n; c += 512)
                for (int i = c + 1; i < c + 512; i++)
                    assertEquals(slotOf[c], slotOf[i], "job at " + c + " index " + i);
        }
    }

    @Test
    void aRegionStopsClaimingAfterAFailure() {
        try (Parallel pool = Parallel.of(4)) {
            AtomicInteger visits = new AtomicInteger();
            assertThrows(
                    RuntimeException.class,
                    () ->
                            pool.loop(
                                    0,
                                    1 << 24,
                                    i -> {
                                        if (visits.incrementAndGet() == 1000)
                                            throw new RuntimeException();
                                    }));
            assertTrue(visits.get() < (1 << 24) / 4, "stopped early: " + visits.get());
        }
    }

    // ---- round three: lifetimes, common patterns, odd submitters ----

    @Test
    void bodyCapturesAreReleasedAfterTheLoop() {
        try (Parallel pool = Parallel.of(4)) {
            WeakReference<Object> captured = runAndDrop(pool);
            for (int attempt = 0; attempt < 50 && captured.get() != null; attempt++) {
                System.gc();
                LockSupport.parkNanos(10_000_000);
            }
            assertTrue(captured.get() == null, "the pool kept the last body's captures alive");
        }
    }

    private static WeakReference<Object> runAndDrop(Parallel pool) {
        Object big = new byte[1 << 20];
        pool.loop(
                0,
                1000,
                i -> {
                    if (big == null) fail();
                });
        return new WeakReference<>(big);
    }

    @Test
    void closeRacingTheFirstLoopLeavesNoThreadBehind() throws Exception {
        int before = jinferThreads();
        for (int round = 0; round < 100; round++) {
            Parallel pool = Parallel.of(4);
            Thread submitter = new Thread(() -> pool.loop(0, 1 << 12, i -> spin(1_000)));
            submitter.start();
            if ((round & 1) == 0) spin(50_000);
            pool.close();
            submitter.join(30_000);
            assertFalse(submitter.isAlive());
        }
        waitUntil(() -> jinferThreads() <= before, 10_000);
    }

    @Test
    void anOpenPoolThatIsDroppedKeepsItsWorkersUntilClosed() throws Exception {
        int before = jinferThreads();
        Parallel[] holder = {Parallel.of(3)};
        holder[0].loop(0, 100, i -> {});
        waitUntil(() -> jinferThreads() == before + 2, 5_000);
        WeakReference<Parallel> ref = new WeakReference<>(holder[0]);
        holder[0] = null;
        for (int attempt = 0; attempt < 20; attempt++) {
            System.gc();
            LockSupport.parkNanos(5_000_000);
        }
        assertTrue(ref.get() != null, "workers reference the pool: it is not collected");
        assertEquals(before + 2, jinferThreads(), "documented: an own pool must be closed");
        ref.get().close();
        waitUntil(() -> jinferThreads() == before, 10_000);
    }

    @Test
    void staticFormsCoverEveryShape() {
        AtomicInteger visits = new AtomicInteger();
        Set<Integer> slots = ConcurrentHashMap.newKeySet();
        Parallel.forLoop(
                3,
                7,
                (i, slot) -> {
                    visits.incrementAndGet();
                    slots.add(slot);
                });
        Parallel.forLoop(
                4,
                (i, slot) -> {
                    visits.incrementAndGet();
                    slots.add(slot);
                });
        Parallel.forLoop(3, 7, i -> visits.incrementAndGet());
        Parallel.forLoop(4, i -> visits.incrementAndGet());
        assertEquals(16, visits.get());
        assertTrue(slots.stream().allMatch(sl -> sl >= 0 && sl < Parallel.threads()));
    }

    @Test
    void aNullBodyIsRejectedUpFront() {
        try (Parallel pool = Parallel.of(2)) {
            assertThrows(NullPointerException.class, () -> pool.loop(0, 0, (Parallel.Job) null));
            assertThrows(NullPointerException.class, () -> pool.loop(5, (Parallel.Job) null));
            assertThrows(NullPointerException.class, () -> pool.loop(0, 10, (IntConsumer) null));
            assertThrows(NullPointerException.class, () -> pool.loop(0, 0, (IntConsumer) null));
            assertThrows(NullPointerException.class, () -> pool.loop(0, (IntConsumer) null));
            assertThrows(NullPointerException.class, () -> Parallel.forLoop(3, (IntConsumer) null));
        }
    }

    @Test
    void perSlotPartialSumsAreAReductionWithoutAtomics() {
        for (int width = 1; width <= 9; width++) {
            try (Parallel pool = Parallel.of(width)) {
                for (int n : new int[] {0, 1, 2, width - 1, width, width + 1, 4 * width, 100_003}) {
                    long[] partial = new long[width];
                    pool.loop(0, n, (i, slot) -> partial[slot] += i);
                    long sum = 0;
                    for (long p : partial) sum += p;
                    assertEquals((long) n * (n - 1) / 2, sum, "width " + width + " n " + n);
                }
            }
        }
    }

    @Test
    void lazyPerSlotScratchIsCreatedExactlyOnce() {
        try (Parallel pool = Parallel.of(6)) {
            AtomicReferenceArray<Object> scratch = new AtomicReferenceArray<>(6);
            AtomicInteger created = new AtomicInteger();
            for (int round = 0; round < 50; round++) {
                pool.loop(
                        0,
                        6 * 32,
                        (i, slot) -> {
                            if (scratch.get(slot) == null) {
                                Object mine = new float[256];
                                if (!scratch.compareAndSet(slot, null, mine))
                                    fail("slot " + slot + " created twice");
                                created.incrementAndGet();
                            }
                            ((float[]) scratch.get(slot))[0] += 1;
                            spin(5_000);
                        });
            }
            assertTrue(created.get() <= 6);
            float total = 0;
            for (int s = 0; s < 6; s++)
                if (scratch.get(s) != null) total += ((float[]) scratch.get(s))[0];
            assertEquals(50 * 6 * 32, (int) total);
        }
    }

    @Test
    void produceThenConsumeIsTwoRegionsNotOneSpin() {
        try (Parallel pool = Parallel.of(4)) {
            int n = 10_000;
            int[] produced = new int[n];
            pool.loop(0, n, i -> produced[i] = i * 3);
            long[] partial = new long[4];
            pool.loop(0, n, (i, slot) -> partial[slot] += produced[i]);
            assertEquals(3L * n * (n - 1) / 2, partial[0] + partial[1] + partial[2] + partial[3]);
        }
    }

    @Test
    void theSubmitterInterruptedMidRegionByAWorkerFinishesWithTheFlagSet() {
        try (Parallel pool = Parallel.of(4)) {
            Thread submitter = Thread.currentThread();
            try {
                AtomicInteger visits = new AtomicInteger();
                pool.loop(
                        0,
                        4 * 4 * 8,
                        (i, slot) -> {
                            if (slot != 0 && i % 8 == 0) submitter.interrupt();
                            spin(100_000);
                            visits.incrementAndGet();
                        });
                assertEquals(128, visits.get());
                assertTrue(submitter.isInterrupted());
            } finally {
                Thread.interrupted();
            }
        }
    }

    @Test
    void workersInterruptedByTheirOwnBodiesKeepServing() {
        try (Parallel pool = Parallel.of(4)) {
            for (int round = 0; round < 20; round++) {
                AtomicInteger visits = new AtomicInteger();
                pool.loop(
                        0,
                        256,
                        (i, slot) -> {
                            if (slot != 0) Thread.currentThread().interrupt();
                            spin(10_000);
                            visits.incrementAndGet();
                        });
                assertEquals(256, visits.get());
                if (round % 5 == 0)
                    LockSupport.parkNanos(2_000_000); // let them park with the flag set
            }
        }
    }

    @Test
    void forkJoinAndParallelStreamThreadsCanSubmit() throws Exception {
        try (Parallel pool = Parallel.of(4)) {
            AtomicInteger visits = new AtomicInteger();
            ForkJoinPool.commonPool()
                    .submit(() -> pool.loop(0, 1000, i -> visits.incrementAndGet()))
                    .get(30, TimeUnit.SECONDS);
            assertEquals(1000, visits.get());
            visits.set(0);
            IntStream.range(0, 16)
                    .parallel()
                    .forEach(s -> pool.loop(0, 1000, i -> visits.incrementAndGet()));
            assertEquals(16_000, visits.get());
        }
    }

    @Test
    void aChainOfPoolsWithoutACycleCompletes() {
        try (Parallel a = Parallel.of(3);
                Parallel b = Parallel.of(3);
                Parallel c = Parallel.of(3);
                Parallel d = Parallel.of(3)) {
            AtomicInteger visits = new AtomicInteger();
            a.loop(
                    0,
                    8,
                    i ->
                            b.loop(
                                    0,
                                    8,
                                    j ->
                                            c.loop(
                                                    0,
                                                    8,
                                                    k ->
                                                            d.loop(
                                                                    0,
                                                                    8,
                                                                    l ->
                                                                            visits
                                                                                    .incrementAndGet()))));
            assertEquals(8 * 8 * 8 * 8, visits.get());
        }
    }

    @Test
    void aFailureInAnotherPoolsRegionPropagatesThroughBoth() {
        try (Parallel a = Parallel.of(3);
                Parallel b = Parallel.of(3)) {
            RuntimeException boom = new RuntimeException("in b");
            RuntimeException thrown =
                    assertThrows(
                            RuntimeException.class,
                            () ->
                                    a.loop(
                                            0,
                                            64,
                                            i ->
                                                    b.loop(
                                                            0,
                                                            64,
                                                            j -> {
                                                                if (i == 10 && j == 20) throw boom;
                                                            })));
            assertSame(boom, thrown);
            AtomicInteger visits = new AtomicInteger();
            a.loop(0, 64, i -> b.loop(0, 64, j -> visits.incrementAndGet()));
            assertEquals(64 * 64, visits.get());
        }
    }

    @Test
    void widthAndInlineBehaviourSurviveClose() {
        Parallel pool = Parallel.of(5);
        pool.close();
        assertEquals(5, pool.width(), "width is a property of the pool, not of its threads");
        long[] partial = new long[5];
        pool.loop(0, 100, (i, slot) -> partial[slot] += i);
        assertEquals(4950, partial[0]);
    }

    @Test
    void idleWorkersParkInsteadOfBurningCpu() throws Exception {
        try (Parallel pool = Parallel.of(4)) {
            pool.loop(0, 100, i -> {});
            Set<String> workers = workerNames(pool);
            waitUntil(() -> allParked(workers), 5_000);
            ThreadMXBean mx = ManagementFactory.getThreadMXBean();
            long[] ids =
                    Thread.getAllStackTraces().keySet().stream()
                            .filter(t -> workers.contains(t.getName()))
                            .mapToLong(Thread::getId)
                            .toArray();
            long cpu0 = 0;
            for (long id : ids) cpu0 += mx.getThreadCpuTime(id);
            LockSupport.parkNanos(200_000_000);
            long cpu1 = 0;
            for (long id : ids) cpu1 += mx.getThreadCpuTime(id);
            assertTrue(
                    cpu1 - cpu0 < 20_000_000,
                    "parked workers used " + (cpu1 - cpu0) / 1e6 + " ms cpu in 200 ms");
        }
    }

    private static int jinferThreads() {
        int n = 0;
        for (Thread t : Thread.getAllStackTraces().keySet())
            if (t.getName().startsWith("jinfer-") && t.isAlive()) n++;
        return n;
    }

    @Test
    void poolsAreTellingInAThreadDump() {
        try (Parallel a = Parallel.of(3);
                Parallel b = Parallel.of(2)) {
            a.loop(0, 100, i -> {});
            b.loop(0, 100, i -> {});
            Set<String> namesA = workerNames(a), namesB = workerNames(b);
            assertEquals(2, namesA.size());
            assertEquals(1, namesB.size());
            assertTrue(namesA.stream().noneMatch(namesB::contains), "distinct names per pool");
            assertTrue(namesA.stream().allMatch(n -> n.startsWith("jinfer-p")));
            assertTrue(
                    a.toString().contains("width=3") && a.toString().contains("workers=2"),
                    a.toString());
            b.close();
            assertTrue(b.toString().contains("closed"), b.toString());
        }
        assertTrue(
                Parallel.shared().toString().startsWith("jinfer["), Parallel.shared().toString());
    }

    @Test
    void aRegionIsOneAllocationAndWorkersAllocateNothing() {
        try (Parallel pool = Parallel.of(4)) {
            ThreadMXBean mx = ManagementFactory.getThreadMXBean();
            if (!(mx instanceof com.sun.management.ThreadMXBean sun)
                    || !sun.isThreadAllocatedMemorySupported()) return;
            int[] sink = new int[64];
            IntConsumer simple = i -> sink[i & 63] = i;
            Parallel.Job body = (i, slot) -> sink[i & 63] = slot;
            for (int r = 0; r < 20_000; r++) {
                pool.loop(0, 64, simple);
                pool.loop(0, 64, body);
            } // warm
            Set<Thread> workers = ConcurrentHashMap.newKeySet();
            pool.loop(0, 64, i -> workers.add(Thread.currentThread()));
            long[] ids =
                    workers.stream()
                            .filter(t -> t != Thread.currentThread())
                            .mapToLong(Thread::getId)
                            .toArray();
            long me = Thread.currentThread().getId();
            int regions = 100_000;
            long selfBefore = sun.getThreadAllocatedBytes(me), workersBefore = 0;
            for (long id : ids) workersBefore += sun.getThreadAllocatedBytes(id);
            for (int r = 0; r < regions; r++) pool.loop(0, 64, simple);
            for (int r = 0; r < regions; r++) pool.loop(0, 64, body);
            long self = sun.getThreadAllocatedBytes(me) - selfBefore, workersAfter = 0;
            for (long id : ids) workersAfter += sun.getThreadAllocatedBytes(id);
            long perRegion = self / (2L * regions);
            assertTrue(perRegion <= 96, "submitter allocates " + perRegion + " bytes per region");
            assertTrue(
                    workersAfter - workersBefore < 64L * 1024,
                    "workers allocated " + (workersAfter - workersBefore) + " bytes");
        }
    }

    // ---- helpers ----

    private static void spin(long nanos) {
        long until = System.nanoTime() + nanos;
        while (System.nanoTime() < until) Thread.onSpinWait();
    }

    private static void await(CountDownLatch latch) {
        await(latch, 30);
    }

    private static boolean await(CountDownLatch latch, int seconds) {
        try {
            return latch.await(seconds, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return false;
        }
    }

    private static void waitUntil(java.util.function.BooleanSupplier condition, long millis) {
        long until = System.nanoTime() + millis * 1_000_000L;
        while (!condition.getAsBoolean()) {
            if (System.nanoTime() > until) fail("condition not met in " + millis + " ms");
            LockSupport.parkNanos(100_000);
        }
    }

    private static Set<String> threadNames() {
        Set<String> names = ConcurrentHashMap.newKeySet();
        for (Thread t : Thread.getAllStackTraces().keySet()) names.add(t.getName());
        return names;
    }

    /** The pool's workers are the jinfer-N threads that did not exist before it ran. */
    private static Set<String> workerNames(Parallel pool) {
        Set<Thread> mine = ConcurrentHashMap.newKeySet();
        for (int round = 0; round < 50; round++)
            pool.loop(
                    0,
                    pool.width() * 4,
                    i -> {
                        mine.add(Thread.currentThread());
                        spin(20_000);
                    });
        Set<String> names = ConcurrentHashMap.newKeySet();
        for (Thread t : mine) if (t != Thread.currentThread()) names.add(t.getName());
        return names;
    }

    private static boolean allParked(Set<String> workers) {
        int parked = 0;
        for (Thread t : Thread.getAllStackTraces().keySet())
            if (workers.contains(t.getName()) && t.getState() == Thread.State.WAITING) parked++;
        return parked == workers.size();
    }

    @SuppressWarnings("unchecked")
    private static <T extends Throwable> void sneaky(Throwable t) throws T {
        throw (T) t;
    }
}
