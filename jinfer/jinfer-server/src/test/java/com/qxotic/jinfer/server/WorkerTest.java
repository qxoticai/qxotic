package com.qxotic.jinfer.server;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import org.junit.jupiter.api.Test;

/**
 * The worker is the server's overload behaviour: one generation at a time, a bounded FIFO in front,
 * and backpressure rather than unbounded pile-up. That path only runs when a server is already in
 * trouble, which is exactly when it must be right - and none of it needs a model.
 */
class WorkerTest {

    /** A job that parks until released, so the queue can be filled deterministically. */
    private record Blocker(CountDownLatch release, CountDownLatch started) implements Runnable {
        @Override
        public void run() {
            started.countDown();
            try {
                release.await(10, TimeUnit.SECONDS);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
    }

    @Test
    void jobsRunInSubmissionOrderOnOneThread() throws Exception {
        Worker worker = new Worker();
        worker.start();
        ConcurrentLinkedQueue<Integer> order = new ConcurrentLinkedQueue<>();
        ConcurrentLinkedQueue<String> threads = new ConcurrentLinkedQueue<>();
        for (int i = 0; i < 4; i++) {
            int n = i;
            assertTrue(
                    worker.submitAndWait(
                            () -> {
                                order.add(n);
                                threads.add(Thread.currentThread().getName());
                            }),
                    "an idle worker must accept");
        }
        assertEquals(List.of(0, 1, 2, 3), List.copyOf(order));
        assertEquals(
                1,
                Set.copyOf(threads).size(),
                "generation must be serialized on ONE thread: " + threads);
    }

    @Test
    void aFullQueueIsRejectedRatherThanQueuedForever() throws Exception {
        Worker worker = new Worker();
        worker.start();
        CountDownLatch release = new CountDownLatch(1);
        CountDownLatch running = new CountDownLatch(1);
        AtomicBoolean rejected = new AtomicBoolean();
        try {
            // one job occupies the worker, then fill the bounded queue behind it
            List<Thread> submitters = new java.util.ArrayList<>();
            Thread first =
                    Thread.ofPlatform()
                            .start(() -> worker.submitAndWait(new Blocker(release, running)));
            submitters.add(first);
            assertTrue(running.await(5, TimeUnit.SECONDS), "the worker never picked up the job");

            for (int i = 0; i < ServerFlags.SERVER_QUEUE + 2; i++) {
                Thread t =
                        Thread.ofPlatform()
                                .start(
                                        () -> {
                                            if (!worker.submitAndWait(
                                                    new Blocker(release, new CountDownLatch(1)))) {
                                                rejected.set(true);
                                            }
                                        });
                submitters.add(t);
            }
            // more submissions than the queue can hold: at least one must be refused, and refused
            // WITHOUT waiting - a caller that blocks forever is the pile-up this exists to prevent
            long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10);
            while (!rejected.get() && System.nanoTime() < deadline) Thread.sleep(20);
            assertTrue(rejected.get(), "a full queue must answer with backpressure");
        } finally {
            release.countDown();
        }
    }

    @Test
    void anIdleWorkerReportsItself() {
        Worker worker = new Worker();
        worker.start();
        assertEquals(0, worker.queueDepth());
        assertFalse(worker.busy());
        assertTrue(Worker.retryAfterSeconds() > 0, "backpressure must suggest when to retry");
    }
}
