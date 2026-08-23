package com.qxotic.jinfer;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.LockSupport;
import java.util.function.IntConsumer;

/**
 * Persistent spin-barrier workers for latency-sensitive decode loops. Workers spin briefly between
 * the many small parallel regions within a token, then park while decode is idle. {@link Parallel}
 * enforces the single-submitter contract.
 */
final class SpinPool {
    /** Roughly 100 microseconds of {@link Thread#onSpinWait()} before an idle worker parks. */
    private static final int SPIN_BEFORE_PARK = 1 << 16;

    private final int participants; // background workers + the submitting thread
    private final Thread[] workers;
    private volatile IntConsumer action; // current region body
    private int rangeStart; // published by the volatile generation store below
    private int rangeEnd;
    private final AtomicInteger arrived = new AtomicInteger();
    private final AtomicInteger parked = new AtomicInteger();
    private volatile long generation;
    private volatile Throwable failure; // a worker's exception, re-thrown to the submitter

    SpinPool(int participants) {
        this.participants = Math.max(1, participants);
        this.workers = new Thread[this.participants - 1];
        for (int w = 0; w < workers.length; w++) {
            final int id = w;
            Thread t = new Thread(() -> workerLoop(id), "jinfer-decode-spin-" + w);
            t.setDaemon(true);
            workers[w] = t;
            t.start();
        }
    }

    private void workerLoop(int id) {
        long seen = 0;
        int idle = 0;
        while (true) {
            if (generation != seen) { // volatile acquire publishes rangeStart/rangeEnd/action
                seen = generation;
                idle = 0;
                try {
                    runSlice(id);
                } catch (Throwable t) {
                    // Keep the worker alive and surface the failure to the submitter at the
                    // barrier.
                    failure = t;
                } finally {
                    arrived.incrementAndGet();
                }
            } else if (idle < SPIN_BEFORE_PARK) {
                idle++;
                Thread.onSpinWait();
            } else {
                // idle a while: park until the next dispatch. Register first, then re-check
                // generation, so a dispatch landing in this window is guaranteed to unpark us
                // (sequentially consistent handshake with parked).
                parked.incrementAndGet();
                if (generation == seen) {
                    LockSupport.park();
                }
                parked.decrementAndGet();
                idle = 0;
            }
        }
    }

    /**
     * Each participant owns one contiguous band. This avoids the cache and memory-bank aliasing
     * seen with strided slices while letting each core stream one sequential weight region.
     */
    private void runSlice(int id) {
        IntConsumer body = action;
        long start = rangeStart, end = rangeEnd;
        long span = (end - start + participants - 1) / participants;
        int lo = (int) (start + id * span);
        int hi = (int) Math.min(end, start + (id + 1L) * span);
        for (int i = lo; i < hi; i++) {
            body.accept(i);
        }
    }

    /**
     * Run {@code action} over [start,end) across the pool; the caller is the final participant and
     * returns once every index has been processed. Caller must be the sole submitter (enforced by
     * Parallel).
     */
    void forLoop(int start, int end, IntConsumer body) {
        long n = (long) end - start;
        if (n <= 0) {
            return;
        }
        if (n == 1 || participants == 1) { // not worth waking the pool
            for (int i = start; i < end; i++) body.accept(i);
            return;
        }
        action = body;
        rangeStart = start;
        rangeEnd = end;
        arrived.set(0);
        generation++; // volatile release: publishes the region + signals workers
        if (parked.get() != 0) { // some worker parked (idle gap) — wake them; ~0 during a token
            for (Thread w : workers) LockSupport.unpark(w);
        }
        try {
            runSlice(participants - 1); // submitter is the last participant (no idle thread)
        } catch (Throwable t) {
            failure = t;
        }
        while (arrived.get() < participants - 1) {
            Thread.onSpinWait();
        }
        Throwable f = failure;
        action = null;
        failure = null;
        if (f != null) { // a participant threw — propagate like ForkJoinPool would
            throw Parallel.unchecked(f);
        }
    }
}
