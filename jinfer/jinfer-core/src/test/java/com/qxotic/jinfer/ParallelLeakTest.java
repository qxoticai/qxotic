package com.qxotic.jinfer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

import java.lang.ref.WeakReference;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.LockSupport;
import org.junit.jupiter.api.Test;

/**
 * Leaks and stuck threads: every path through the API runs under a watchdog that fails with a
 * thread dump instead of hanging the build, and is bracketed by a census of the pool's threads
 * (none may survive {@code close()}), a heap check (a closed pool is collectable, a live pool
 * retains nothing from past regions) and a stuck-worker check (after any region, failed or not,
 * every worker is back to spinning or parked).
 */
class ParallelLeakTest {

    private static final long WATCHDOG_SECONDS = 60;

    // ---- threads ----

    @Test
    void everyPathReleasesItsThreadsOnClose() {
        int before = poolThreads();
        String[] paths = {
            "idle", "after-loops", "after-failure", "while-parked", "while-spinning",
            "while-running", "after-nested", "after-other-pool", "width-1", "closed-twice"
        };
        for (String path : paths) {
            watchdog(path, () -> runPath(path));
            waitUntil(path + ": threads released", () -> poolThreads() <= before, 10_000);
        }
    }

    private static void runPath(String path) {
        Parallel pool = Parallel.of(path.equals("width-1") ? 1 : 4);
        switch (path) {
            case "idle" -> {}
            case "after-loops" -> {
                for (int r = 0; r < 100; r++) pool.loop(0, 1000, i -> {});
            }
            case "after-failure" -> {
                for (int r = 0; r < 20; r++)
                    assertThrows(
                            RuntimeException.class,
                            () ->
                                    pool.loop(
                                            0,
                                            1000,
                                            i -> {
                                                if (i % 7 == 3) throw new RuntimeException();
                                            }));
            }
            case "while-parked" -> {
                pool.loop(0, 1000, i -> {});
                LockSupport.parkNanos(5_000_000); // past SPIN_NANOS: the workers are parked
            }
            case "while-spinning" -> pool.loop(0, 1000, i -> {}); // closed within 100 us: spinning
            case "while-running" -> {
                CountDownLatch started = new CountDownLatch(1);
                Thread t =
                        new Thread(
                                () ->
                                        pool.loop(
                                                0,
                                                400,
                                                i -> {
                                                    started.countDown();
                                                    spin(200_000);
                                                }));
                t.start();
                await(started);
                pool.close();
                join(t);
            }
            case "after-nested" -> pool.loop(0, 64, i -> pool.loop(0, 64, j -> {}));
            case "after-other-pool" -> {
                try (Parallel other = Parallel.of(3)) {
                    pool.loop(0, 64, i -> other.loop(0, 64, j -> {}));
                }
            }
            case "width-1" -> pool.loop(0, 1000, i -> {});
            case "closed-twice" -> {
                pool.loop(0, 10, i -> {});
                pool.close();
            }
            default -> fail(path);
        }
        pool.close();
        pool.loop(0, 10, i -> {}); // still usable, inline
    }

    @Test
    void rapidCreateLoopCloseCyclesLeaveNothing() {
        int before = poolThreads();
        watchdog(
                "cycles",
                () -> {
                    for (int r = 0; r < 500; r++) {
                        try (Parallel pool = Parallel.of(1 + (r % 8))) {
                            pool.loop(0, 64 + r, i -> {});
                            if (r % 3 == 0)
                                assertThrows(
                                        RuntimeException.class,
                                        () ->
                                                pool.loop(
                                                        0,
                                                        64,
                                                        i -> {
                                                            throw new RuntimeException();
                                                        }));
                        }
                    }
                });
        waitUntil("500 pools released", () -> poolThreads() <= before, 15_000);
    }

    @Test
    void workersAreNeverStuckAfterARegion() {
        try (Parallel pool = Parallel.of(4)) {
            pool.loop(0, 1000, i -> {});
            Set<Thread> workers = workersOf(pool);
            assertEquals(3, workers.size());
            Runnable[] regions = {
                () -> pool.loop(0, 10_000, i -> {}),
                () ->
                        assertThrows(
                                RuntimeException.class,
                                () ->
                                        pool.loop(
                                                0,
                                                10_000,
                                                i -> {
                                                    throw new RuntimeException();
                                                })),
                () ->
                        assertThrows(
                                StackOverflowError.class,
                                () ->
                                        pool.loop(
                                                0,
                                                64,
                                                (i, s) -> {
                                                    if (s != 0) overflow(0);
                                                    spin(50_000);
                                                })),
                () -> pool.loop(0, 64, i -> pool.loop(0, 64, j -> spin(1_000))),
                () ->
                        assertThrows(
                                Error.class,
                                () ->
                                        pool.loop(
                                                0,
                                                64,
                                                i -> {
                                                    throw new Error();
                                                })),
            };
            for (int round = 0; round < 10; round++) {
                for (Runnable region : regions) {
                    watchdog("region " + round, region);
                    waitUntil(
                            "workers idle",
                            () -> workers.stream().allMatch(ParallelLeakTest::idle),
                            5_000);
                    for (Thread w : workers)
                        assertTrue(w.isAlive(), "a worker died: " + w.getName());
                }
            }
        }
    }

    @Test
    void theLockIsReleasedAfterAFailure() throws Exception {
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
            Thread other = new Thread(() -> pool.loop(0, 100, i -> visits.incrementAndGet()));
            other.start();
            other.join(10_000);
            assertTrue(!other.isAlive(), "another submitter blocked after a failed region");
            assertEquals(100, visits.get());
        }
    }

    // ---- memory ----

    @Test
    void aClosedPoolIsCollectable() {
        WeakReference<Parallel> ref = makeUseAndClose();
        for (int attempt = 0; attempt < 50 && ref.get() != null; attempt++) {
            System.gc();
            LockSupport.parkNanos(10_000_000);
        }
        assertTrue(ref.get() == null, "closed pool still reachable: a worker or static holds it");
    }

    private static WeakReference<Parallel> makeUseAndClose() {
        Parallel pool = Parallel.of(4);
        pool.loop(0, 10_000, i -> spin(100));
        pool.close();
        return new WeakReference<>(pool);
    }

    @Test
    void aLivePoolRetainsNothingFromPastRegions() {
        try (Parallel pool = Parallel.of(4)) {
            WeakReference<Object>[] captures = runCapturing(pool, 20);
            for (int attempt = 0; attempt < 50; attempt++) {
                boolean allGone = true;
                for (WeakReference<Object> c : captures) allGone &= c.get() == null;
                if (allGone) break;
                System.gc();
                LockSupport.parkNanos(10_000_000);
            }
            int retained = 0;
            for (WeakReference<Object> c : captures) if (c.get() != null) retained++;
            assertEquals(0, retained, "bodies' captures retained after their regions");
        }
    }

    @SuppressWarnings("unchecked")
    private static WeakReference<Object>[] runCapturing(Parallel pool, int n) {
        WeakReference<Object>[] refs = new WeakReference[n];
        for (int r = 0; r < n; r++) {
            Object payload = new byte[256 << 10];
            if (r % 2 == 0)
                pool.loop(
                        0,
                        1000,
                        i -> {
                            if (payload == null) fail();
                        });
            else
                assertThrows(
                        RuntimeException.class,
                        () ->
                                pool.loop(
                                        0,
                                        1000,
                                        i -> {
                                            if (payload != null) throw new RuntimeException();
                                        }));
            refs[r] = new WeakReference<>(payload);
        }
        return refs;
    }

    @Test
    void aHundredThousandRegionsDoNotGrowTheHeap() {
        try (Parallel pool = Parallel.of(4)) {
            for (int r = 0; r < 10_000; r++) pool.loop(0, 8, i -> {});
            long base = usedAfterGc();
            for (int r = 0; r < 100_000; r++) pool.loop(0, 8, i -> {});
            long after = usedAfterGc();
            assertTrue(after - base < 8L << 20, "heap grew by " + ((after - base) >> 10) + " KiB");
        }
    }

    @Test
    void theSharedPoolRetainsNothingEither() {
        WeakReference<Object> captured = runOnShared();
        for (int attempt = 0; attempt < 50 && captured.get() != null; attempt++) {
            System.gc();
            LockSupport.parkNanos(10_000_000);
        }
        assertTrue(captured.get() == null, "the shared pool kept a body's captures");
    }

    private static WeakReference<Object> runOnShared() {
        Object payload = new byte[1 << 20];
        Parallel.forLoop(
                0,
                1000,
                i -> {
                    if (payload == null) fail();
                });
        return new WeakReference<>(payload);
    }

    // ---- hangs ----

    @Test
    void nothingHangsUnderAdversarialBodies() {
        try (Parallel pool = Parallel.of(4)) {
            Runnable[] scenarios = {
                // the submitter is the only one to throw
                () ->
                        assertThrows(
                                RuntimeException.class,
                                () ->
                                        pool.loop(
                                                0,
                                                4096,
                                                (i, s) -> {
                                                    if (s == 0) throw new RuntimeException();
                                                    spin(1_000);
                                                })),
                // every worker throws, the submitter never does
                () ->
                        assertThrows(
                                RuntimeException.class,
                                () ->
                                        pool.loop(
                                                0,
                                                4096,
                                                (i, s) -> {
                                                    if (s != 0) throw new RuntimeException();
                                                    spin(1_000);
                                                })),
                // the last index throws
                () ->
                        assertThrows(
                                RuntimeException.class,
                                () ->
                                        pool.loop(
                                                0,
                                                4096,
                                                i -> {
                                                    if (i == 4095) throw new RuntimeException();
                                                })),
                // the first index throws
                () ->
                        assertThrows(
                                RuntimeException.class,
                                () ->
                                        pool.loop(
                                                0,
                                                4096,
                                                i -> {
                                                    if (i == 0) throw new RuntimeException();
                                                })),
                // a body interrupts everyone it can see
                () ->
                        pool.loop(
                                0,
                                256,
                                i -> {
                                    for (Thread t : workersOf(pool)) t.interrupt();
                                    Thread.currentThread().interrupt();
                                }),
                // a body closes and reopens nothing: closes the pool it runs on, twice
                () -> {
                    Parallel p = Parallel.of(4);
                    p.loop(
                            0,
                            256,
                            i -> {
                                p.close();
                                p.close();
                                spin(1_000);
                            });
                },
                // deep nesting with failures at the bottom
                () ->
                        assertThrows(
                                RuntimeException.class,
                                () ->
                                        pool.loop(
                                                0,
                                                8,
                                                i ->
                                                        pool.loop(
                                                                0,
                                                                8,
                                                                j ->
                                                                        pool.loop(
                                                                                0,
                                                                                8,
                                                                                k -> {
                                                                                    if (k == 7)
                                                                                        throw new RuntimeException();
                                                                                })))),
                // a body that sleeps longer than the spin budget on every participant
                () -> pool.loop(0, 8, i -> LockSupport.parkNanos(1_000_000)),
                // a body that submits to the shared pool while the shared pool is busy elsewhere
                () -> {
                    Thread busy = new Thread(() -> Parallel.forLoop(0, 100_000, i -> spin(500)));
                    busy.start();
                    pool.loop(0, 64, i -> Parallel.forLoop(0, 64, j -> {}));
                    join(busy);
                },
            };
            for (int round = 0; round < 3; round++) {
                for (int s = 0; s < scenarios.length; s++) {
                    watchdog("scenario " + s + " round " + round, scenarios[s]);
                    Thread.interrupted();
                }
            }
        }
    }

    @Test
    void manySubmittersHammeringFailingLoopsFinish() {
        try (Parallel pool = Parallel.of(4)) {
            watchdog(
                    "hammer",
                    () -> {
                        Thread[] threads = new Thread[8];
                        AtomicReference<Throwable> failure = new AtomicReference<>();
                        for (int t = 0; t < threads.length; t++) {
                            int id = t;
                            threads[t] =
                                    new Thread(
                                            () -> {
                                                try {
                                                    for (int r = 0; r < 200; r++) {
                                                        if ((r + id) % 2 == 0)
                                                            assertThrows(
                                                                    RuntimeException.class,
                                                                    () ->
                                                                            pool.loop(
                                                                                    0,
                                                                                    500,
                                                                                    i -> {
                                                                                        if (i
                                                                                                == 250)
                                                                                            throw new RuntimeException();
                                                                                    }));
                                                        else pool.loop(0, 500, i -> spin(100));
                                                    }
                                                } catch (Throwable e) {
                                                    failure.compareAndSet(null, e);
                                                }
                                            });
                            threads[t].start();
                        }
                        for (Thread t : threads) join(t);
                        if (failure.get() != null) throw new AssertionError(failure.get());
                    });
        }
    }

    // ---- helpers ----

    /** Runs {@code task} on its own thread; a timeout fails with a dump of every pool thread. */
    private static void watchdog(String what, Runnable task) {
        AtomicReference<Throwable> failure = new AtomicReference<>();
        Thread t =
                new Thread(
                        () -> {
                            try {
                                task.run();
                            } catch (Throwable e) {
                                failure.set(e);
                            }
                        },
                        "watchdog-" + what);
        t.start();
        try {
            t.join(TimeUnit.SECONDS.toMillis(WATCHDOG_SECONDS));
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        if (t.isAlive()) {
            StringBuilder dump =
                    new StringBuilder(what + " hung after " + WATCHDOG_SECONDS + " s\n");
            for (Map.Entry<Thread, StackTraceElement[]> e : Thread.getAllStackTraces().entrySet()) {
                if (!e.getKey().getName().startsWith("jinfer") && e.getKey() != t) continue;
                dump.append("  ")
                        .append(e.getKey().getName())
                        .append(' ')
                        .append(e.getKey().getState())
                        .append('\n');
                for (StackTraceElement frame : e.getValue())
                    dump.append("    at ").append(frame).append('\n');
            }
            fail(dump.toString());
        }
        if (failure.get() != null) throw new AssertionError(what, failure.get());
    }

    private static int poolThreads() {
        int n = 0;
        for (Thread t : Thread.getAllStackTraces().keySet())
            if (t.getName().startsWith("jinfer-p") && t.isAlive()) n++;
        return n;
    }

    private static Set<Thread> workersOf(Parallel pool) {
        String prefix = pool.toString().substring(0, pool.toString().indexOf('[')) + "-";
        Set<Thread> workers = ConcurrentHashMap.newKeySet();
        for (Thread t : Thread.getAllStackTraces().keySet())
            if (t.getName().startsWith(prefix)) workers.add(t);
        return workers;
    }

    private static boolean idle(Thread worker) {
        Thread.State state = worker.getState();
        if (state == Thread.State.WAITING) return true; // parked
        StackTraceElement[] frames = worker.getStackTrace();
        for (StackTraceElement f : frames) if (f.getMethodName().equals("work")) return false;
        return state == Thread.State.RUNNABLE; // spinning between regions
    }

    private static long usedAfterGc() {
        for (int i = 0; i < 3; i++) {
            System.gc();
            LockSupport.parkNanos(20_000_000);
        }
        Runtime rt = Runtime.getRuntime();
        return rt.totalMemory() - rt.freeMemory();
    }

    private static void waitUntil(
            String what, java.util.function.BooleanSupplier condition, long millis) {
        long until = System.nanoTime() + millis * 1_000_000L;
        while (!condition.getAsBoolean()) {
            if (System.nanoTime() > until) fail(what + ": not within " + millis + " ms");
            LockSupport.parkNanos(200_000);
        }
    }

    private static void await(CountDownLatch latch) {
        try {
            if (!latch.await(30, TimeUnit.SECONDS)) fail("latch");
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    private static void join(Thread t) {
        try {
            t.join(TimeUnit.SECONDS.toMillis(WATCHDOG_SECONDS));
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        if (t.isAlive()) fail(t.getName() + " did not finish");
    }

    private static void spin(long nanos) {
        long until = System.nanoTime() + nanos;
        while (System.nanoTime() < until) Thread.onSpinWait();
    }

    private static int overflow(int depth) {
        return overflow(depth + 1) + 1;
    }
}
