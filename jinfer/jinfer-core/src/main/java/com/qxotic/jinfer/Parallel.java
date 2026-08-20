package com.qxotic.jinfer;

import com.qxotic.jinfer.telemetry.PerformanceCliff;
import java.util.concurrent.Callable;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.IntConsumer;
import java.util.function.Supplier;
import java.util.stream.IntStream;

/**
 * jinfer's work runners, and the reason there are two. {@link #parallelFor} fans out on the common
 * pool - right for the compute-bound prefill gemm. A decode step runs on a spin-barrier pool at
 * physical-core width instead: decode is memory-bandwidth bound, so a second SMT sibling only
 * contends for its core's load/store ports, and the persistent workers dispatch the ~360 tiny
 * regions a single token fires without building a task tree or parking between them.
 */
public final class Parallel {
    public static void parallelFor(int startInclusive, int endExclusive, IntConsumer action) {
        if (SPIN_OWNER.get()
                == Thread.currentThread()) { // inside the active decode step: spin-barrier pool
            DECODE_SPIN.parallelFor(startInclusive, endExclusive, action);
            return;
        }
        IntStream.range(startInclusive, endExclusive).parallel().forEach(action);
    }

    public static void forRows(int rows, IntConsumer action) {
        if (rows == 1) {
            action.accept(0);
        } else {
            parallelFor(0, rows, action);
        }
    }

    // Single-token (decode) work is memory-bandwidth bound: one thread per PHYSICAL core already
    // saturates
    // DRAM, and a second SMT sibling on the same core only contends for its load/store ports and
    // L1, cutting
    // effective bandwidth. The shared common pool sizes to every logical CPU — right for the
    // compute-bound
    // prefill gemm, wrong for decode. So a decode step runs at physical-core width on the SpinPool,
    // whose
    // persistent workers dispatch with no task-tree allocation and no park/unpark between the ~360
    // tiny
    // regions a token fires (the dominant cost vs llama.cpp). The SpinPool takes one submitter at a
    // time, so
    // concurrent decodes (a multi-request server) fall back to the equivalent-width ForkJoinPool.
    private static final SpinPool DECODE_SPIN = new SpinPool(RuntimeFlags.DECODE_THREADS);
    private static final ForkJoinPool DECODE_POOL = new ForkJoinPool(RuntimeFlags.DECODE_THREADS);
    // The thread running the active decode step (DECODE_SPIN's sole submitter): its nested
    // parallelFor calls
    // route to the spin pool; every other thread's keep the common pool. null = no decode step in
    // flight.
    private static final AtomicReference<Thread> SPIN_OWNER = new AtomicReference<>();

    /**
     * Evaluate a memory-bandwidth-bound step (one decode token's forward / logits) at physical-core
     * width: on the SpinPool when it is free (the single active decode), else the ForkJoinPool.
     * Unchecked exceptions from {@code step} propagate unchanged.
     */
    public static <T> T onDecodePool(Supplier<T> step) {
        if (RuntimeFlags.DECODE_SPIN) {
            if (SPIN_OWNER.compareAndSet(null, Thread.currentThread())) {
                try {
                    return step.get();
                } finally {
                    SPIN_OWNER.set(null);
                }
            }
            // The CAS was lost to another decode. (Spin explicitly disabled takes the same pool
            // silently - that was the caller's choice, not a cliff.)
            PerformanceCliff.DECODE_CONTENTION.report();
        }
        return DECODE_POOL
                .submit((Callable<T>) step::get)
                .join(); // concurrent decode or spin disabled
    }
}
