// Persistent spin-barrier worker pool for the latency-sensitive decode path. A decode token fires ~360
// tiny parallel regions (one per matmul / norm / sampling pass); the shared ForkJoinPool allocates a task
// tree and PARKS workers between regions, so each region pays unpark + task-tree latency that, on the
// bandwidth-bound decode gemv, roughly HALVES effective DRAM bandwidth (GemvSeqBench: 62 -> 125 GB/s here).
// These workers instead SPIN (Thread.onSpinWait) on a generation counter, so a dispatch is two plain stores
// + one volatile bump — no allocation, no park/unpark. Between tokens (idle past SPIN_BEFORE_PARK) they park
// so the cores aren't burned when no decode is running. Single-submitter only — guarded by Parallel.
// llama.cpp's graph executor works the same way (persistent threads, spin-then-futex barriers).
package com.qxotic.jinfer;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.LockSupport;
import java.util.function.IntConsumer;

final class SpinPool {
    private static final int SPIN_BEFORE_PARK = 1 << 16;   // ~100us of onSpinWait before an idle worker parks

    private final int participants;             // background workers + the submitting thread
    private final Thread[] workers;
    private volatile IntConsumer action;        // current region body
    private int rangeStart, rangeEnd;           // published by the volatile generation store below
    private final AtomicInteger arrived = new AtomicInteger();
    private final AtomicInteger parked = new AtomicInteger();
    private volatile long generation;
    private volatile boolean shutdown;
    private volatile Throwable failure;         // a worker's exception, re-thrown to the submitter

    SpinPool(int participants) {
        this.participants = Math.max(1, participants);
        this.workers = new Thread[this.participants - 1];
        for (int w = 0; w < workers.length; w++) {
            final int id = w;
            Thread t = new Thread(() -> workerLoop(id), "decode-spin-" + w);
            t.setDaemon(true);
            workers[w] = t;
            t.start();
        }
    }

    private void workerLoop(int id) {
        long seen = 0;
        int idle = 0;
        while (true) {
            if (shutdown) return;
            if (generation != seen) {              // volatile acquire publishes rangeStart/rangeEnd/action
                seen = generation;
                idle = 0;
                try {
                    runSlice(id);
                } catch (Throwable t) {       // keep the worker alive; surface to the submitter at the barrier
                    failure = t;
                } finally {
                    arrived.incrementAndGet();
                }
            } else if (idle < SPIN_BEFORE_PARK) {
                idle++;
                Thread.onSpinWait();
            } else {
                // idle a while: park until the next dispatch. Register first, then re-check generation, so a
                // dispatch landing in this window is guaranteed to unpark us (seq-cst handshake with parked).
                parked.incrementAndGet();
                if (generation == seen && !shutdown) {
                    LockSupport.park();
                }
                parked.decrementAndGet();
                idle = 0;
            }
        }
    }

    /** Participant {@code id} owns a single CONTIGUOUS band of the index range. Measured strictly better than
     *  a strided {id, id+P, ...} split across model shapes (Qwen 3.5 4B/2B, gemma-4 MoE, Llama-3.2): the
     *  strided stride aliases DRAM banks / cache sets for some weight row-strides (a −5% Qwen regression that
     *  contiguous bands turn into +5–7%), while each core here streams one sequential weight region. */
    private void runSlice(int id) {
        IntConsumer act = action;
        int start = rangeStart, end = rangeEnd;
        int span = (end - start + participants - 1) / participants;
        int lo = start + id * span, hi = Math.min(end, lo + span);
        for (int i = lo; i < hi; i++) {
            act.accept(i);
        }
    }

    /** Run {@code action} over [start,end) across the pool; the caller is the final participant and returns
     *  once every index has been processed. Caller must be the sole submitter (enforced by Parallel). */
    void parallelFor(int start, int end, IntConsumer act) {
        int n = end - start;
        if (n <= 0) {
            return;
        }
        if (n == 1 || participants == 1) {     // not worth waking the pool
            if (TRACE) trInline++;
            for (int i = start; i < end; i++) act.accept(i);
            return;
        }
        long t0 = TRACE ? System.nanoTime() : 0;
        action = act;
        rangeStart = start;
        rangeEnd = end;
        arrived.set(0);
        generation++;                          // volatile release: publishes the region + signals workers
        if (parked.get() != 0) {               // some worker parked (idle gap) — wake them; ~0 during a token
            for (Thread w : workers) LockSupport.unpark(w);
        }
        try {
            runSlice(participants - 1);        // submitter is the last participant (no idle thread)
        } catch (Throwable t) {
            failure = t;
        }
        long tSlice = TRACE ? System.nanoTime() : 0;
        while (arrived.get() < participants - 1) {
            Thread.onSpinWait();
        }
        if (TRACE) { long e = System.nanoTime(); trCalls++; trDispatchNs += e - t0; trBarrierNs += e - tSlice; }
        Throwable f = failure;
        if (f != null) {                       // a participant threw — propagate like ForkJoinPool would
            failure = null;
            sneakyThrow(f);
        }
    }

    // --- dispatch overhead trace (-Djinfer.spinTrace) ---
    static final boolean TRACE = System.getProperty("jinfer.spinTrace") != null;
    private static long trCalls, trInline, trDispatchNs, trBarrierNs;
    static void traceReport(String tag, long tokens) {
        if (!TRACE || tokens <= 0) return;
        System.err.printf("[spin %s] %d tok: parallelFor=%.1f/tok inline=%.1f/tok  inDispatch=%.3fms/tok  barrierWait=%.3fms/tok (%.0f%% of dispatch)%n",
                tag, tokens, (double) trCalls / tokens, (double) trInline / tokens,
                trDispatchNs / 1e6 / tokens, trBarrierNs / 1e6 / tokens, 100.0 * trBarrierNs / Math.max(1, trDispatchNs));
    }

    @SuppressWarnings("unchecked")
    private static <E extends Throwable> void sneakyThrow(Throwable e) throws E {
        throw (E) e;
    }
}
