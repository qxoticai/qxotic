package com.qxotic.jinfer.server;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;

class WorkerTest {

    @Test
    void serializesJobs() {
        try (Worker worker = new Worker(2)) {
            worker.start();
            var order = new ConcurrentLinkedQueue<Integer>();
            var threads = new ConcurrentLinkedQueue<String>();
            for (int i = 0; i < 4; i++) {
                int n = i;
                assertEquals(
                        Worker.Result.COMPLETED,
                        worker.submitAndWait(
                                () -> {
                                    order.add(n);
                                    threads.add(Thread.currentThread().getName());
                                }));
            }
            assertEquals(List.of(0, 1, 2, 3), List.copyOf(order));
            assertEquals(1, Set.copyOf(threads).size());
        }
    }

    @Test
    void rejectsBeyondTheBound() throws Exception {
        try (Worker worker = new Worker(2)) {
            worker.start();
            CountDownLatch running = new CountDownLatch(1);
            CountDownLatch release = new CountDownLatch(1);
            List<Thread> submitters = new ArrayList<>();
            submitters.add(
                    Thread.ofPlatform()
                            .start(
                                    () ->
                                            worker.submitAndWait(
                                                    () -> {
                                                        running.countDown();
                                                        await(release);
                                                    })));
            assertTrue(running.await(5, TimeUnit.SECONDS));

            for (int i = 0; i < 2; i++) {
                submitters.add(
                        Thread.ofPlatform()
                                .start(() -> worker.submitAndWait(() -> await(release))));
            }
            long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
            while (worker.queued() < 2 && System.nanoTime() < deadline) Thread.onSpinWait();
            assertEquals(Worker.Result.FULL, worker.submitAndWait(() -> {}));
            release.countDown();
            for (Thread thread : submitters) thread.join(5_000);
        }
    }

    @Test
    void closeReleasesQueuedSubmitters() throws Exception {
        Worker worker = new Worker(1);
        worker.start();
        CountDownLatch running = new CountDownLatch(1);
        AtomicReference<Worker.Result> queued = new AtomicReference<>();
        Thread first =
                Thread.ofPlatform()
                        .start(
                                () ->
                                        worker.submitAndWait(
                                                () -> {
                                                    running.countDown();
                                                    try {
                                                        Thread.sleep(60_000);
                                                    } catch (InterruptedException ignored) {
                                                        Thread.currentThread().interrupt();
                                                    }
                                                }));
        assertTrue(running.await(5, TimeUnit.SECONDS));
        Thread second = Thread.ofPlatform().start(() -> queued.set(worker.submitAndWait(() -> {})));
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
        while (worker.queued() != 1 && System.nanoTime() < deadline) Thread.onSpinWait();

        worker.close();
        second.join(5_000);
        first.join(5_000);
        assertFalse(second.isAlive(), "queued caller was stranded");
        assertEquals(Worker.Result.CLOSED, queued.get());
        assertEquals(Worker.Result.CLOSED, worker.submitAndWait(() -> {}));
    }

    @Test
    void interruptRemovesAQueuedJob() throws Exception {
        try (Worker worker = new Worker(1)) {
            worker.start();
            CountDownLatch running = new CountDownLatch(1);
            CountDownLatch release = new CountDownLatch(1);
            Thread first =
                    Thread.ofPlatform()
                            .start(
                                    () ->
                                            worker.submitAndWait(
                                                    () -> {
                                                        running.countDown();
                                                        await(release);
                                                    }));
            assertTrue(running.await(5, TimeUnit.SECONDS));
            AtomicBoolean executed = new AtomicBoolean();
            AtomicReference<Worker.Result> result = new AtomicReference<>();
            Thread queued =
                    Thread.ofPlatform()
                            .start(
                                    () ->
                                            result.set(
                                                    worker.submitAndWait(
                                                            () -> executed.set(true))));
            long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
            while (worker.queued() != 1 && System.nanoTime() < deadline) Thread.onSpinWait();

            queued.interrupt();
            queued.join(5_000);
            release.countDown();
            first.join(5_000);

            assertEquals(Worker.Result.INTERRUPTED, result.get());
            assertFalse(executed.get());
            assertEquals(0, worker.queued());
        }
    }

    @Test
    void interruptDoesNotAbandonAnActiveJob() throws Exception {
        try (Worker worker = new Worker(1)) {
            worker.start();
            CountDownLatch running = new CountDownLatch(1);
            CountDownLatch release = new CountDownLatch(1);
            AtomicReference<Worker.Result> result = new AtomicReference<>();
            Thread submitter =
                    Thread.ofPlatform()
                            .start(
                                    () ->
                                            result.set(
                                                    worker.submitAndWait(
                                                            () -> {
                                                                running.countDown();
                                                                await(release);
                                                            })));
            assertTrue(running.await(5, TimeUnit.SECONDS));

            submitter.interrupt();
            submitter.join(50);
            assertTrue(submitter.isAlive(), "active response must retain its handler");

            release.countDown();
            submitter.join(5_000);
            assertEquals(Worker.Result.COMPLETED, result.get());
            assertTrue(submitter.isInterrupted());
        }
    }

    private static void await(CountDownLatch latch) {
        try {
            latch.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
}
