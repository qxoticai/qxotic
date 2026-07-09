package com.qxotic.jinfer;

import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.SynchronousQueue;

/**
 * The single generation worker. Generation runs one request at a time on a dedicated thread fed by
 * a bounded FIFO queue ({@code llama.serverQueue}; 0 = reject unless idle): a fixed serialization
 * point so the inference state is never shared across requests, with backpressure instead of
 * unbounded pile-up. Handlers parse/validate on their own thread and only block here, so a fixed
 * HTTP pool also caps the threads a slow client can pin.
 */
final class Worker {

    private final BlockingQueue<Runnable> queue =
            RuntimeFlags.SERVER_QUEUE == 0
                    ? new SynchronousQueue<>()
                    : new ArrayBlockingQueue<>(RuntimeFlags.SERVER_QUEUE);
    private volatile boolean busy;

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
                                    System.err.println("generation worker:");
                                    t.printStackTrace();
                                }
                            }
                        });
    }

    /**
     * Submits and blocks until the job completes; throws if the queue is full. For startup work
     * (prompt-cache warming) that must finish before the server serves requests.
     */
    void runToCompletion(Runnable job) {
        if (!submitAndWait(job)) {
            throw new IllegalStateException("generation queue full at startup");
        }
    }

    /**
     * Submits {@code job} and waits for it; returns false (without waiting) when the queue is full,
     * so the caller can answer with backpressure.
     */
    boolean submitAndWait(Runnable job) {
        CountDownLatch done = new CountDownLatch(1);
        if (!queue.offer(
                () -> {
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

    /** Retry-After seconds suggested when the queue is full. */
    static int retryAfterSeconds() {
        return Math.max(1, 2 * (RuntimeFlags.SERVER_QUEUE + 1));
    }
}
