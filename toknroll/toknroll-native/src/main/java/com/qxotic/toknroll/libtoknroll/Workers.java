package com.qxotic.toknroll.libtoknroll;

import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.LockSupport;
import java.util.function.IntConsumer;
import java.util.stream.IntStream;
import org.graalvm.nativeimage.ProcessProperties;
import org.graalvm.nativeimage.c.function.CFunction;
import org.graalvm.word.PointerBase;

/**
 * The threads a batch runs on: one per core, caller included, started by the first parallel batch, each
 * waiting for work in C between jobs, where a fork finds it idle (toknroll_isolate.c says why not
 * the common pool). Where C has no workers (Windows, libjinfer) a batch runs on the common pool; in
 * a forked child, which has none of the threads, on the calling thread.
 */
final class Workers {
    /** The threads a batch is shared between, the caller's included. */
    static final int THREADS = Runtime.getRuntime().availableProcessors();

    @CFunction("toknroll__workers")
    private static native int available();

    @CFunction("toknroll__worker_new")
    private static native PointerBase newWorker();

    @CFunction("toknroll__worker_wait")
    private static native long await(PointerBase worker, long seen);

    @CFunction("toknroll__workers_wake")
    private static native void wake();

    /** One batch's tasks, claimed one at a time by its caller and the workers. */
    private static final class Job {
        final int tasks;
        final IntConsumer task;
        final Thread caller = Thread.currentThread();
        final AtomicInteger next = new AtomicInteger();
        final AtomicInteger done = new AtomicInteger();
        volatile Throwable failure;

        Job(int tasks, IntConsumer task) {
            this.tasks = tasks;
            this.task = task;
        }

        void work() {
            for (int i; (i = next.getAndIncrement()) < tasks; ) {
                try {
                    task.accept(i);
                } catch (Throwable t) {
                    if (failure == null) failure = t;
                }
                if (done.incrementAndGet() == tasks) LockSupport.unpark(caller);
            }
        }
    }

    private static final Queue<Job> JOBS = new ConcurrentLinkedQueue<>();

    /** The process the runtime started in. */
    private static final long PROCESS = ProcessProperties.getProcessID();

    private static boolean started; // guarded by Workers.class

    /** Runs every task, sharing work across cores only when there is more than one task. */
    static void run(int tasks, IntConsumer task) {
        if (tasks <= 1 || ProcessProperties.getProcessID() != PROCESS) {
            for (int i = 0; i < tasks; i++) task.accept(i);
            return;
        }
        if (available() == 0) {
            IntStream.range(0, tasks).parallel().forEach(task);
            return;
        }
        start();
        Job job = new Job(tasks, task);
        JOBS.add(job);
        wake();
        job.work();
        while (job.done.get() < tasks) LockSupport.park(job);
        JOBS.remove(job);
        Throwable failure = job.failure;
        if (failure instanceof RuntimeException e) throw e;
        if (failure instanceof Error e) throw e;
        if (failure != null) throw new IllegalStateException(failure);
    }

    /**
     * The workers, once. Each is listed with the gate before this returns, so a fork from then on
     * waits for it; until then the caller, inside its call, holds the fork back.
     */
    private static synchronized void start() {
        if (started) return;
        started = true;
        AtomicInteger listed = new AtomicInteger(); // all a worker keeps of its start
        for (int i = 1; i < THREADS; i++) {
            Thread worker =
                    new Thread(
                            () -> {
                                PointerBase self = newWorker();
                                listed.incrementAndGet();
                                // ponytail: calloc failed: it ends here, a few frames outside
                                // the gate, in a process that is out of memory anyway
                                if (self.isNonNull()) loop(self);
                            },
                            "toknroll-batch-" + i);
            worker.setDaemon(true);
            worker.start();
        }
        // ponytail: yields for the threads' start, once per process; nothing to wake, nothing kept
        while (listed.get() < THREADS - 1) Thread.yield();
    }

    /** A worker's life: wait in C, then work on whatever jobs are queued. */
    private static void loop(PointerBase self) {
        long seen = 0;
        for (; ; ) {
            seen = await(self, seen);
            for (Job job; (job = JOBS.peek()) != null; ) {
                job.work();
                JOBS.remove(job);
            }
        }
    }

    private Workers() {}
}
