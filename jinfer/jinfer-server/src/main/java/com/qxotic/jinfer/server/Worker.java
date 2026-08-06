package com.qxotic.jinfer.server;

import com.qxotic.jinfer.*;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.SynchronousQueue;

/**
 * The single generation worker. Generation runs one request at a time on a dedicated thread fed by
 * a bounded FIFO queue ({@code queueDepth}; 0 = reject unless idle): a fixed serialization point so
 * the inference state is never shared across requests, with backpressure instead of unbounded
 * pile-up. Handlers parse/validate on their own thread and only block here, so a fixed HTTP pool
 * also caps the threads a slow client can pin.
 */
final class Worker {

    private final BlockingQueue<Runnable> queue;
    private volatile boolean busy;

    Worker(int queueDepth) {
        this.queue =
                queueDepth == 0 ? new SynchronousQueue<>() : new ArrayBlockingQueue<>(queueDepth);
    }

    void start() {
        Thread.ofPlatform()
                .name("generation-worker")
                .daemon(true)
                .start(
                        () -> {
                            while (true) {
                                try {
                                    Runnable job = queue.take();
                                    busy = true;
                                    try {
                                        job.run();
                                    } finally {
                                        busy = false;
                                    }
                                } catch (InterruptedException e) {
                                    return;
                                } catch (Throwable t) {
                                    Log.LOG.log(System.Logger.Level.ERROR, "generation worker", t);
                                }
                            }
                        });
    }

    /**
     * Submits {@code job} and waits for it; returns false (without waiting) when the queue is full,
     * so the caller can answer with backpressure.
     */
    boolean submitAndWait(Runnable job) {
        CountDownLatch done = new CountDownLatch(1);
        long queuedAt = System.nanoTime();
        if (!queue.offer(
                () -> {
                    // published on the worker thread, which is also the thread the generation and
                    // its telemetry event run on - so the wait lands on the right request
                    com.qxotic.jinfer.telemetry.Telemetry.queueWait(System.nanoTime() - queuedAt);
                    try {
                        job.run();
                    } finally {
                        done.countDown();
                    }
                })) {
            return false;
        }
        await(done);
        return true;
    }

    private static void await(CountDownLatch done) {
        try {
            done.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    int queueDepth() {
        return queue.size();
    }

    boolean busy() {
        return busy;
    }
}
