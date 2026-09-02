package com.qxotic.jam;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.LockSupport;

/**
 * A test pool with real threads and unique slots: the caller is slot 0, worker k is slot k. Spins
 * briefly between regions, then parks. For the backends' own tests and benches; jinfer has its own.
 */
public final class TestPool implements JAM.Parallel {

    public static JAM.Parallel of(int width) {
        return width <= 1 ? JAM.Parallel.INLINE : new TestPool(width);
    }

    private final Worker[] workers;
    private volatile Region current = new Region(0, null);

    private TestPool(int width) {
        workers = new Worker[width - 1];
        for (int i = 0; i < workers.length; i++) {
            workers[i] = new Worker(i + 1);
            workers[i].start();
        }
    }

    @Override
    public int width() {
        return workers.length + 1;
    }

    @Override
    public synchronized void run(int count, Job body) {
        if (count <= 0) return;
        Region region = new Region(count, body);
        current = region;
        for (Worker w : workers) if (w.parked) LockSupport.unpark(w);
        region.work(0);
        while (region.finished.get() < count) Thread.onSpinWait();
        Throwable failure = region.failure;
        if (failure instanceof RuntimeException e) throw e;
        if (failure instanceof Error e) throw e;
        if (failure != null) throw new RuntimeException(failure);
    }

    private static final class Region {
        final int count;
        final Job body;
        final AtomicInteger next = new AtomicInteger(), finished = new AtomicInteger();
        volatile Throwable failure;

        Region(int count, Job body) {
            this.count = count;
            this.body = body;
        }

        void work(int slot) {
            for (int i; (i = next.getAndIncrement()) < count; ) {
                try {
                    body.run(i, slot);
                } catch (Throwable t) {
                    failure = t;
                }
                finished.incrementAndGet();
            }
        }
    }

    private final class Worker extends Thread {
        final int slot;
        volatile boolean parked;

        Worker(int slot) {
            this.slot = slot;
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
                    r.work(slot);
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
