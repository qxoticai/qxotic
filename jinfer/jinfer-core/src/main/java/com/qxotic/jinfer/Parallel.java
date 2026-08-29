package com.qxotic.jinfer;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.LockSupport;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.IntConsumer;

/**
 * Jinfer's one thread pool: {@link RuntimeFlags#THREADS} participants, the calling thread being one
 * of them. Every parallel loop in the engine and in the jam backends runs here, so the thread
 * budget is structural - there is no second pool to spin, park or starve against.
 *
 * <p>A loop is a region: participants claim contiguous chunks of it from a counter (dynamic, so a
 * slow chunk never holds the region), the caller works alongside the workers and returns when the
 * last chunk is done. Between regions the workers spin {@link #SPIN_NANOS} - the many small regions
 * of one decode token or one prefill matmul are microseconds apart - and park when the engine is
 * idle. Regions are serialized: a second thread submitting (a second session) waits for the running
 * region, which is the fair share on a machine the first session already fills. A loop submitted
 * from inside a region runs inline on that thread; loops are independent and non-blocking, so that
 * is always correct, only less parallel.
 */
public final class Parallel {

    /** Spin this long for the next region before parking. */
    private static final long SPIN_NANOS = 100_000L;

    /** Chunks per participant per region: enough for the tail to balance. */
    private static final int CHUNKS_PER_PARTICIPANT = 4;

    private static final ReentrantLock LOCK = new ReentrantLock();

    /** The current region; workers act on every new object they see here. */
    private static volatile Region current = new Region(0, 0, null, null);

    private static volatile Worker[] workers;

    private Parallel() {}

    /** The compute thread budget: how many participants a region has. */
    public static int threads() {
        return RuntimeFlags.THREADS;
    }

    /** {@code body.accept(i)} for every {@code i < count}. */
    public static void forLoop(int count, IntConsumer body) {
        forLoop(0, count, body);
    }

    /**
     * {@code body.accept(i)} for every {@code i} in {@code [start, end)}. Iterations may run
     * concurrently and in any order; they must be independent and non-blocking.
     */
    public static void forLoop(int start, int end, IntConsumer body) {
        if (start >= end) return;
        Thread me = Thread.currentThread();
        Region running = current;
        boolean nested = me instanceof Worker || (running.submitter == me && !running.done());
        if (end - start == 1 || RuntimeFlags.THREADS == 1 || nested) {
            for (int i = start; i < end; i++) body.accept(i);
            return;
        }
        Worker[] pool = pool();
        LOCK.lock();
        try {
            Region region = new Region(start, end, body, me);
            current = region;
            for (Worker w : pool) if (w.parked) LockSupport.unpark(w);
            region.work();
            while (!region.done()) Thread.onSpinWait();
            if (region.failure != null) throw unchecked(region.failure);
        } finally {
            LOCK.unlock();
        }
    }

    private static Worker[] pool() {
        Worker[] p = workers;
        if (p == null) {
            synchronized (Parallel.class) {
                p = workers;
                if (p == null) {
                    p = new Worker[RuntimeFlags.THREADS - 1];
                    for (int i = 0; i < p.length; i++) {
                        p[i] = new Worker(i);
                        p[i].start();
                    }
                    workers = p;
                }
            }
        }
        return p;
    }

    /** One loop: chunks are claimed from {@code next}; {@code finished} counts done indices. */
    private static final class Region {
        final int start, end, chunk;
        final IntConsumer body;
        final Thread submitter;
        final AtomicInteger next;
        final AtomicInteger finished = new AtomicInteger();
        volatile Throwable failure;

        Region(int start, int end, IntConsumer body, Thread submitter) {
            this.start = start;
            this.end = end;
            this.body = body;
            this.submitter = submitter;
            this.next = new AtomicInteger(start);
            this.chunk =
                    Math.max(1, (end - start) / (RuntimeFlags.THREADS * CHUNKS_PER_PARTICIPANT));
        }

        boolean done() {
            return finished.get() >= end - start;
        }

        void work() {
            for (int lo; (lo = next.getAndAdd(chunk)) < end; ) {
                int hi = Math.min(end, lo + chunk);
                try {
                    for (int i = lo; i < hi; i++) body.accept(i);
                } catch (Throwable thrown) {
                    if (failure == null) failure = thrown;
                }
                finished.addAndGet(hi - lo);
            }
        }
    }

    private static final class Worker extends Thread {
        volatile boolean parked;

        Worker(int index) {
            super("jinfer-" + index);
            setDaemon(true);
        }

        @Override
        public void run() {
            Region seen = current;
            long idleSince = System.nanoTime();
            while (true) {
                Region region = current;
                if (region != seen) {
                    seen = region;
                    region.work();
                    idleSince = System.nanoTime();
                } else if (System.nanoTime() - idleSince < SPIN_NANOS) {
                    Thread.onSpinWait();
                } else {
                    parked = true;
                    if (current == seen) LockSupport.park(this);
                    parked = false;
                    idleSince = System.nanoTime();
                }
            }
        }
    }

    static RuntimeException unchecked(Throwable failure) {
        if (failure instanceof RuntimeException runtime) return runtime;
        if (failure instanceof Error error) throw error;
        return new RuntimeException(failure);
    }
}
