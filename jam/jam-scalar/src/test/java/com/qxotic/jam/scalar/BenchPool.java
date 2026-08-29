package com.qxotic.jam.scalar;

import com.qxotic.jam.JAM;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.LockSupport;
import java.util.function.IntConsumer;

/**
 * A {@link JAM.Parallel} for the tests and benches: the calling thread plus {@code width - 1}
 * spinning workers (park after 20 us idle), regions serialized. What jinfer's pool does, without
 * jinfer.
 */
final class BenchPool implements JAM.Parallel {

    static JAM.Parallel of(int width) {
        return width <= 1 ? JAM.Parallel.INLINE : new BenchPool(width);
    }

    private final Worker[] workers;
    private volatile Region current = new Region(0, null);

    private BenchPool(int width) {
        workers = new Worker[width - 1];
        for (int i = 0; i < workers.length; i++) {
            workers[i] = new Worker();
            workers[i].start();
        }
    }

    @Override
    public int width() {
        return workers.length + 1;
    }

    @Override
    public synchronized void forLoop(int count, IntConsumer body) {
        if (count <= 0) return;
        Region region = new Region(count, body);
        current = region;
        for (Worker w : workers) if (w.parked) LockSupport.unpark(w);
        region.work();
        while (region.finished.get() < count) Thread.onSpinWait();
        if (region.failure != null) throw new RuntimeException(region.failure);
    }

    private static final class Region {
        final int count;
        final IntConsumer body;
        final AtomicInteger next = new AtomicInteger(), finished = new AtomicInteger();
        volatile Throwable failure;

        Region(int count, IntConsumer body) {
            this.count = count;
            this.body = body;
        }

        void work() {
            for (int i; (i = next.getAndIncrement()) < count; ) {
                try {
                    body.accept(i);
                } catch (Throwable t) {
                    failure = t;
                }
                finished.incrementAndGet();
            }
        }
    }

    private final class Worker extends Thread {
        volatile boolean parked;

        Worker() {
            setDaemon(true);
        }

        @Override
        public void run() {
            Region seen = current;
            long idle = System.nanoTime();
            while (true) {
                Region r = current;
                if (r != seen) {
                    seen = r;
                    r.work();
                    idle = System.nanoTime();
                } else if (System.nanoTime() - idle < 20_000L) {
                    Thread.onSpinWait();
                } else {
                    parked = true;
                    if (current == seen) LockSupport.park(this);
                    parked = false;
                    idle = System.nanoTime();
                }
            }
        }
    }
}
