package com.qxotic.jinfer;

import com.qxotic.jinfer.telemetry.PerformanceCliff;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.RecursiveAction;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.IntConsumer;
import java.util.function.Supplier;

/** Routes finite, non-blocking loops to Jinfer-owned compute and decode workers. */
public final class Parallel {
    private static final long WORKER_KEEP_ALIVE_SECONDS = 60;
    private static final int TASKS_PER_WORKER = 4;
    private static final ForkJoinPool COMPUTE_POOL = newPool(RuntimeFlags.COMPUTE_THREADS);
    private static final ForkJoinPool DECODE_POOL = newPool(RuntimeFlags.DECODE_THREADS);
    private static final AtomicReference<Thread> ACTIVE_SPIN_SUBMITTER = new AtomicReference<>();

    /**
     * Run {@code action} once for every index in {@code [startInclusive, endExclusive)}.
     *
     * <p>Iterations may run concurrently and in any order. Nested calls are supported; iterations
     * must be independent and non-blocking.
     */
    public static void forLoop(int startInclusive, int endExclusive, IntConsumer action) {
        if (startInclusive >= endExclusive) {
            return;
        }
        if ((long) endExclusive - startInclusive == 1) {
            action.accept(startInclusive);
            return;
        }
        if (ACTIVE_SPIN_SUBMITTER.get() == Thread.currentThread()) {
            DecodeSpin.POOL.forLoop(startInclusive, endExclusive, action);
            return;
        }

        ForkJoinPool current = ForkJoinTask.getPool();
        ForkJoinPool pool =
                current == COMPUTE_POOL || current == DECODE_POOL ? current : COMPUTE_POOL;
        LoopTask task =
                new LoopTask(
                        startInclusive,
                        endExclusive,
                        maxChunkSize(pool, startInclusive, endExclusive),
                        action);
        if (current == pool) {
            task.invoke();
        } else {
            pool.invoke(task);
        }
        if (task.failure != null) throw unchecked(task.failure);
    }

    /**
     * Run {@code action} once for every index in {@code [0, count)} under the same execution rules
     * as {@link #forLoop(int, int, IntConsumer)}.
     */
    public static void forLoop(int count, IntConsumer action) {
        forLoop(0, count, action);
    }

    /**
     * Evaluate one memory-bandwidth-bound decode step. The uncontended path uses the low-latency
     * spin pool; concurrent decodes use the bounded decode pool.
     */
    public static <T> T runDecodeStep(Supplier<T> step) {
        if (RuntimeFlags.DECODE_SPIN) {
            if (ACTIVE_SPIN_SUBMITTER.compareAndSet(null, Thread.currentThread())) {
                try {
                    return step.get();
                } finally {
                    ACTIVE_SPIN_SUBMITTER.set(null);
                }
            }
            PerformanceCliff.DECODE_CONTENTION.report();
        }
        // Capture the outcome rather than trusting FJP's channels: an external join() surfaces a
        // REFLECTIVE COPY of the failure (ForkJoinTask.getException) - mangled message, lost
        // identity. quietlyJoin's happens-before makes plain fields safe; the contended path may
        // allocate, the spin path above may not.
        Outcome<T> outcome = new Outcome<>();
        DECODE_POOL
                .submit(
                        () -> {
                            try {
                                outcome.value = step.get();
                            } catch (Throwable t) {
                                outcome.failure = t;
                            }
                        })
                .quietlyJoin();
        if (outcome.failure != null) throw unchecked(outcome.failure); // the original instance
        return outcome.value;
    }

    /** Carrier for the contended decode path - FJP's own result channels mangle exceptions. */
    private static final class Outcome<T> {
        T value;
        Throwable failure;
    }

    private static ForkJoinPool newPool(int threads) {
        // Structured kernel tasks need no compensation workers. Joins may reduce the active count,
        // but the pool never exceeds the configured width.
        return new ForkJoinPool(
                threads,
                ForkJoinPool.defaultForkJoinWorkerThreadFactory,
                null,
                false,
                0,
                threads,
                0,
                null,
                WORKER_KEEP_ALIVE_SECONDS,
                TimeUnit.SECONDS);
    }

    private static int maxChunkSize(ForkJoinPool pool, int start, int end) {
        long size = (long) end - start;
        long tasks = Math.min(size, (long) pool.getParallelism() * TASKS_PER_WORKER);
        return (int) ((size + tasks - 1) / tasks);
    }

    private static final class LoopTask extends RecursiveAction {
        private final int start;
        private final int end;
        private final int maxChunkSize;
        private final IntConsumer action;
        private LoopTask next;
        private Throwable failure;

        private LoopTask(int start, int end, int maxChunkSize, IntConsumer action) {
            this.start = start;
            this.end = end;
            this.maxChunkSize = maxChunkSize;
            this.action = action;
        }

        @Override
        protected void compute() {
            int lo = start;
            int hi = end;
            LoopTask forks = null;
            try {
                while ((long) hi - lo > maxChunkSize) {
                    int middle = lo + (int) (((long) hi - lo) / 2);
                    LoopTask right = new LoopTask(middle, hi, maxChunkSize, action);
                    right.next = forks;
                    forks = right;
                    right.fork();
                    hi = middle;
                }
                for (int i = lo; i < hi; i++) action.accept(i);
            } catch (Throwable thrown) {
                failure = thrown;
            }

            while (forks != null) {
                forks.quietlyJoin();
                if (forks.failure != null) {
                    if (failure == null) failure = forks.failure;
                    else if (failure != forks.failure) failure.addSuppressed(forks.failure);
                }
                forks = forks.next;
            }
        }
    }

    static RuntimeException unchecked(Throwable failure) {
        if (failure instanceof RuntimeException runtime) {
            return runtime;
        }
        if (failure instanceof Error error) {
            throw error;
        }
        return new RuntimeException(failure);
    }

    private static final class DecodeSpin {
        private static final SpinPool POOL = new SpinPool(RuntimeFlags.DECODE_THREADS);
    }

    private Parallel() {}
}
