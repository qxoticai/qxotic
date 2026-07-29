// Producer-consumer audio stream: one float[] chunk per sentence.
// Implements Iterable so you can for-each, and Closeable so you can cancel mid-stream.
//
//   try (AudioStream as = tts.stream("Hello world.", 1.0, 0.667, 42)) {
//       for (float[] chunk : as) {
//           writeToPipe(chunk);    // playback, encoding, broadcast — whatever
//       }
//   }
//
// Async variant — synthesis thread runs in background, caller pulls via hasNext():
//
//   try (AudioStream as = tts.streamAsync("Long text.", 1.0, 0.667, 42)) {
//       for (float[] chunk : as) {  // next() blocks until chunk N+1 is ready
//           play(chunk);
//       }
//   }
//
// close() cancels the producer thread immediately — useful for user interrupts.
package com.qxotic.jinfer.models.inflect2;

import java.io.Closeable;
import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.atomic.AtomicBoolean;

/** A closeable, cancelable, single-consumption iterator over synthesized audio chunks. */
public final class AudioStream implements Iterable<float[]>, Closeable {

    private final Iterator<float[]> iterator;
    private final Runnable cancel;
    private final AtomicBoolean closed = new AtomicBoolean();
    private volatile Iterator<float[]> cachedIterator;

    private AudioStream(Iterator<float[]> iterator, Runnable cancel) {
        this.iterator = iterator;
        this.cancel = cancel;
    }

    /** Cancel the producer and release resources. Safe to call multiple times. */
    @Override
    public void close() {
        if (closed.compareAndSet(false, true)) {
            cancel.run();
        }
    }

    /**
     * Returns a single-use iterator. Subsequent calls return the same instance — this stream can
     * only be consumed once.
     */
    @Override
    public Iterator<float[]> iterator() {
        if (cachedIterator != null) return cachedIterator;
        AtomicBoolean done = new AtomicBoolean();
        cachedIterator =
                new Iterator<>() {
                    private float[] next;

                    @Override
                    public boolean hasNext() {
                        if (next != null) return true;
                        if (done.get()) return false;
                        syncNext();
                        if (next == null) {
                            done.set(true);
                            return false;
                        }
                        return true;
                    }

                    @Override
                    public float[] next() {
                        if (!hasNext()) throw new NoSuchElementException();
                        float[] chunk = next;
                        next = null;
                        return chunk;
                    }

                    private void syncNext() {
                        if (closed.get()) return;
                        try {
                            if (iterator.hasNext()) next = iterator.next();
                        } catch (RuntimeException e) {
                            if (e.getCause() instanceof InterruptedException) {
                                Thread.currentThread().interrupt();
                            }
                        }
                    }
                };
        return cachedIterator;
    }

    // ── factory methods ───────────────────────────────────────────────

    /**
     * Synchronous stream — the iterator calls into the model on the calling thread. Each chunk is
     * synthesized on-demand when next() is called. Close() is a no-op (nothing to cancel).
     */
    static AudioStream sync(Iterator<float[]> chunks) {
        return new AudioStream(chunks, () -> {});
    }

    /**
     * Async stream — a background daemon thread synthesizes chunks into a bounded queue. The
     * calling thread pulls from the queue via the iterator. Chunk N+1 is synthesized while chunk N
     * is being consumed. close() interrupts the producer thread.
     */
    static AudioStream async(BlockingQueue<float[]> q, Thread producer, int chunkCount) {
        Iterator<float[]> it =
                new Iterator<>() {
                    int remaining = chunkCount;

                    @Override
                    public boolean hasNext() {
                        return remaining > 0;
                    }

                    @Override
                    public float[] next() {
                        remaining--;
                        try {
                            return q.take();
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                            throw new NoSuchElementException("interrupted");
                        }
                    }
                };
        return new AudioStream(
                it,
                () -> {
                    producer.interrupt();
                    q.clear();
                });
    }
}
