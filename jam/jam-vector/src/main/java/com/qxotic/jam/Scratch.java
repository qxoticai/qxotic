package com.qxotic.jam;

import java.util.concurrent.ConcurrentLinkedQueue;

/**
 * Per-context F32 dequant scratch pool for the band kernels (Q4_K/Q5_K/Q6_K, MXFP4, NVFP4).
 *
 * <p>Owned by the matmul context (one per {@link VectorJAM} instance, or one shared by jinfer's legacy tensor
 * path) and passed into each {@code gemm} — NOT a {@code static}/{@code ThreadLocal}. This is the deliberate
 * fix for the old {@code BandGemm.DEQUANT_BAND} ThreadLocal, whose buffers were rooted in the (commonPool,
 * JVM-lifetime) worker threads and so were never released when a context was dropped. Here the buffers are
 * reachable only through this object: reused across every {@code mm} call (no per-call allocation in steady
 * state), and collected with the context that owns the pool.
 *
 * <p>A worker {@link #acquire}s a buffer at the top of its slice and {@link #release}s it at the end; the pool
 * is a lock-free free-list, so concurrent band workers within one gemm each get their own buffer. The pool
 * retains at most the peak number of concurrent workers' buffers (≤ {@code THREADS}), each grown to the
 * largest {@code k} seen — a few MB at most, freed when the owning context is GC'd.
 */
public final class Scratch {

    private final ConcurrentLinkedQueue<float[]> pool = new ConcurrentLinkedQueue<>();

    /** A float[] of length ≥ {@code need}, reused from the pool when one fits, else freshly sized. */
    float[] acquire(int need) {
        float[] b;
        while ((b = pool.poll()) != null) {
            if (b.length >= need) return b;   // discard undersized buffers; a right-sized one replaces them
        }
        return new float[need];
    }

    /** Return a buffer for reuse by a later {@code acquire} (on this or a subsequent gemm). */
    void release(float[] b) {
        pool.offer(b);
    }
}
