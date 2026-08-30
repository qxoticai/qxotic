package com.qxotic.jinfer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicIntegerArray;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;

/**
 * Randomized runs against an oracle: random widths, ranges, bodies that spin, throw, nest on the
 * same or another pool, from one or many submitters at once. Every loop must visit each index at
 * most once, and exactly once when nothing threw, surface one of the thrown instances, keep slots
 * in range and unique among the loop's live tasks, and run nested loops of its own pool inline on
 * the same thread and slot. The seed is printed; re-run with -Djinfer.test.seed=N.
 */
class ParallelFuzzTest {

    private static final long SEED = Long.getLong("jinfer.test.seed", System.nanoTime());
    private static final int ROUNDS = Integer.getInteger("jinfer.test.fuzzRounds", 150);

    @Test
    void singleSubmitter() {
        Random random = new Random(SEED);
        System.out.println("ParallelFuzzTest.singleSubmitter seed=" + SEED);
        for (int round = 0; round < ROUNDS; round++) {
            int width = 1 + random.nextInt(9);
            try (Parallel pool = Parallel.of(width);
                    Parallel other = Parallel.of(1 + random.nextInt(4))) {
                Oracle oracle = new Oracle(pool, other, width, new Random(random.nextLong()));
                for (int loop = 0; loop < 20; loop++) oracle.runOneLoop();
            }
        }
    }

    @Test
    void manySubmitters() throws Exception {
        Random random = new Random(SEED ^ 0x5eed);
        System.out.println("ParallelFuzzTest.manySubmitters seed=" + SEED);
        for (int round = 0; round < ROUNDS / 10; round++) {
            int width = 2 + random.nextInt(7);
            int submitters = 2 + random.nextInt(5);
            try (Parallel pool = Parallel.of(width);
                    Parallel other = Parallel.of(1 + random.nextInt(4))) {
                AtomicReference<Throwable> failure = new AtomicReference<>();
                List<Thread> threads = new ArrayList<>();
                for (int s = 0; s < submitters; s++) {
                    Oracle oracle = new Oracle(pool, other, width, new Random(random.nextLong()));
                    Thread t =
                            new Thread(
                                    () -> {
                                        try {
                                            for (int loop = 0; loop < 15; loop++)
                                                oracle.runOneLoop();
                                        } catch (Throwable e) {
                                            failure.compareAndSet(null, e);
                                        }
                                    },
                                    "fuzz-submitter-" + s);
                    threads.add(t);
                    t.start();
                }
                for (Thread t : threads) {
                    t.join(TimeUnit.MINUTES.toMillis(2));
                    if (t.isAlive()) fail("submitter hung, seed=" + SEED);
                }
                if (failure.get() != null) throw new AssertionError("seed=" + SEED, failure.get());
            }
        }
    }

    /** One submitter's random loops on {@code pool}; slots are unique within one loop. */
    private static final class Oracle {
        final Parallel pool, other;
        final int width;
        final Random random;
        final AtomicIntegerArray busy;

        Oracle(Parallel pool, Parallel other, int width, Random random) {
            this.pool = pool;
            this.other = other;
            this.width = width;
            this.random = random;
            this.busy = new AtomicIntegerArray(width);
        }

        void runOneLoop() {
            int start = random.nextInt(50) - 25;
            int count =
                    random.nextInt(6) == 0
                            ? random.nextInt(3)
                            : 1 + random.nextInt(1 + random.nextInt(2000));
            int end = start + count;
            double throwRate = random.nextInt(4) == 0 ? random.nextDouble() * 0.2 : 0;
            int nestRate = random.nextInt(5); // 0 = never
            boolean nestOther = random.nextBoolean();
            int spinMax = random.nextInt(3) == 0 ? 20_000 : 0;
            AtomicIntegerArray visits = new AtomicIntegerArray(Math.max(count, 0));
            Set<Throwable> thrown = ConcurrentHashMap.newKeySet();
            Random bodyRandom = new Random(random.nextLong());
            long seedForBodies = bodyRandom.nextLong();
            Thread submitter = Thread.currentThread();
            boolean outerIsRegion =
                    count >= 2 && width > 1; // else inline, and nested loops are regions
            Set<Thread> threads = ConcurrentHashMap.newKeySet();
            try {
                pool.loop(
                        start,
                        end,
                        (i, slot) -> {
                            threads.add(Thread.currentThread());
                            if (slot < 0 || slot >= width) fail("slot " + slot + " seed=" + SEED);
                            if (Thread.currentThread() == submitter && slot != 0)
                                fail("submitter slot " + slot);
                            if (!busy.compareAndSet(slot, 0, 1))
                                fail("slot " + slot + " busy, seed=" + SEED);
                            try {
                                visits.incrementAndGet(i - start);
                                Random r = new Random(seedForBodies ^ i);
                                if (spinMax > 0) spin(r.nextInt(spinMax));
                                if (r.nextInt(50) == 0) Thread.yield();
                                if (r.nextInt(200) == 0)
                                    Parallel.forLoop(1 + r.nextInt(8), j -> {});
                                if (nestRate > 0 && r.nextInt(nestRate * 4) == 0) {
                                    Thread me = Thread.currentThread();
                                    int[] inner = {0};
                                    if (nestOther) {
                                        other.loop(0, 1 + r.nextInt(40), j -> inner[0]++);
                                        // ponytail: other pool's slots are its own; only the count
                                        // is checked
                                    } else {
                                        pool.loop(
                                                0,
                                                1 + r.nextInt(40),
                                                (j, innerSlot) -> {
                                                    if (outerIsRegion) {
                                                        assertSame(me, Thread.currentThread());
                                                        assertEquals(slot, innerSlot);
                                                    }
                                                    inner[0]++;
                                                });
                                    }
                                    if (inner[0] < 1) fail("nested loop skipped work");
                                }
                                if (throwRate > 0 && r.nextDouble() < throwRate) {
                                    RuntimeException e = new RuntimeException("index " + i);
                                    thrown.add(e);
                                    throw e;
                                }
                            } finally {
                                busy.set(slot, 0);
                            }
                        });
                if (!thrown.isEmpty())
                    fail("loop swallowed " + thrown.size() + " failures, seed=" + SEED);
            } catch (RuntimeException e) {
                if (!thrown.contains(e))
                    throw new AssertionError("foreign exception, seed=" + SEED, e);
            }
            for (int i = 0; i < count; i++) {
                int v = visits.get(i);
                if (v > 1) fail("index " + (start + i) + " visited " + v + "x, seed=" + SEED);
                if (v == 0 && thrown.isEmpty())
                    fail("index " + (start + i) + " skipped without a failure, seed=" + SEED);
            }
            assertTrue(threads.size() <= width, "over budget, seed=" + SEED);
        }

        private static void spin(long nanos) {
            long until = System.nanoTime() + nanos;
            while (System.nanoTime() < until) Thread.onSpinWait();
        }
    }
}
