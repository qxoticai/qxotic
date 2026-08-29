package com.qxotic.jinfer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.fail;

import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicIntegerArray;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;

class ParallelTest {

    @Test
    void forLoopVisitsEveryIndexExactlyOnce() {
        AtomicIntegerArray visits = new AtomicIntegerArray(10_000);
        Set<Thread> threads = ConcurrentHashMap.newKeySet();
        Parallel.forLoop(
                0,
                visits.length(),
                i -> {
                    visits.incrementAndGet(i);
                    threads.add(Thread.currentThread());
                });
        for (int i = 0; i < visits.length(); i++) assertEquals(1, visits.get(i), "index " + i);
        assertEquals(true, threads.size() <= Parallel.threads(), "at most the thread budget");
    }

    @Test
    void singleElementRangeRunsInline() {
        Thread[] seen = new Thread[1];
        Parallel.forLoop(3, 4, i -> seen[0] = Thread.currentThread());
        assertSame(Thread.currentThread(), seen[0]);
    }

    @Test
    void emptyRangesDoNoWork() {
        Parallel.forLoop(4, 4, ignored -> fail("empty range ran"));
        Parallel.forLoop(5, 4, ignored -> fail("reversed range ran"));
    }

    @Test
    void nestedLoopsRunInlineAndVisitEverything() {
        AtomicIntegerArray visits = new AtomicIntegerArray(64);
        Parallel.forLoop(
                0,
                8,
                row -> Parallel.forLoop(0, 8, column -> visits.incrementAndGet(row * 8 + column)));
        for (int i = 0; i < visits.length(); i++) assertEquals(1, visits.get(i), "cell " + i);
    }

    @Test
    void saturatedNestedLoopsComplete() {
        int threads = Parallel.threads();
        int outerCount = threads * 4, innerCount = 256;
        AtomicLong visits = new AtomicLong();
        for (int repetition = 0; repetition < 100; repetition++) {
            AtomicInteger entered = new AtomicInteger();
            Parallel.forLoop(
                    outerCount,
                    ignored -> {
                        if (entered.getAndIncrement() < threads)
                            while (entered.get() < threads) Thread.onSpinWait();
                        Parallel.forLoop(innerCount, inner -> visits.incrementAndGet());
                    });
        }
        assertEquals(100L * outerCount * innerCount, visits.get());
    }

    @Test
    void failuresPropagateTheOriginalInstanceAndWaitForSiblings() {
        IllegalStateException boom = new IllegalStateException("boom");
        AtomicInteger done = new AtomicInteger();
        IllegalStateException thrown =
                assertThrows(
                        IllegalStateException.class,
                        () ->
                                Parallel.forLoop(
                                        0,
                                        64,
                                        i -> {
                                            if (i == 7) throw boom;
                                            done.incrementAndGet();
                                        }));
        assertSame(boom, thrown);
        assertEquals(63, done.get(), "every other index still ran");
        // the pool is usable afterwards
        AtomicInteger again = new AtomicInteger();
        Parallel.forLoop(0, 100, i -> again.incrementAndGet());
        assertEquals(100, again.get());
    }
}
