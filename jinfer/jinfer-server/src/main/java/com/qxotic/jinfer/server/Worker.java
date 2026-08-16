package com.qxotic.jinfer.server;

import com.qxotic.jinfer.telemetry.Telemetry;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.SynchronousQueue;

/** One model, one generation thread, and a bounded FIFO in front of it. */
final class Worker implements AutoCloseable {

    enum Result {
        COMPLETED,
        FULL,
        CLOSED,
        INTERRUPTED
    }

    private final BlockingQueue<Job> queue;
    private volatile boolean busy;
    private volatile boolean closed;
    private Thread thread;

    Worker(int queueCapacity) {
        queue =
                queueCapacity == 0
                        ? new SynchronousQueue<>()
                        : new ArrayBlockingQueue<>(queueCapacity);
    }

    synchronized void start() {
        if (thread != null) return;
        if (closed) throw new IllegalStateException("worker is closed");
        thread = Thread.ofPlatform().name("generation-worker").daemon(true).start(this::run);
    }

    private void run() {
        try {
            while (!closed) {
                Job job = queue.take();
                busy = true;
                try {
                    job.run();
                } catch (Throwable t) {
                    Log.LOG.log(System.Logger.Level.ERROR, "generation worker", t);
                } finally {
                    busy = false;
                    job.finish(true);
                }
            }
        } catch (InterruptedException ignored) {
            Thread.currentThread().interrupt();
        } finally {
            Job job;
            while ((job = queue.poll()) != null) job.finish(false);
        }
    }

    Result submitAndWait(Runnable work) {
        Job job = new Job(work);
        synchronized (this) {
            if (closed) return Result.CLOSED;
            if (!queue.offer(job)) return Result.FULL;
        }
        try {
            job.done.await();
            return job.completed ? Result.COMPLETED : Result.CLOSED;
        } catch (InterruptedException e) {
            if (queue.remove(job)) {
                job.finish(false);
                Thread.currentThread().interrupt();
                return Result.INTERRUPTED;
            }
            // The worker already owns this response. Let that one writer finish instead of
            // returning to a handler that would race it with a second response on the exchange.
            while (job.done.getCount() != 0) {
                try {
                    job.done.await();
                } catch (InterruptedException ignored) {
                    // Preserve the signal below; there is still no safe way to cancel inference.
                }
            }
            Thread.currentThread().interrupt();
            return job.completed ? Result.COMPLETED : Result.CLOSED;
        }
    }

    int queued() {
        return queue.size();
    }

    boolean busy() {
        return busy;
    }

    @Override
    public synchronized void close() {
        if (closed) return;
        closed = true;
        if (thread != null) thread.interrupt();
        Job job;
        while ((job = queue.poll()) != null) job.finish(false);
    }

    private static final class Job implements Runnable {
        private final Runnable work;
        private final CountDownLatch done = new CountDownLatch(1);
        private final long queuedAt = System.nanoTime();
        private volatile boolean completed;

        private Job(Runnable work) {
            this.work = work;
        }

        @Override
        public void run() {
            Telemetry.queueWait(System.nanoTime() - queuedAt);
            work.run();
        }

        private void finish(boolean completed) {
            this.completed = completed;
            done.countDown();
        }
    }
}
