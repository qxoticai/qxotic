package com.qxotic.jam;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.concurrent.ConcurrentLinkedQueue;

/**
 * Per-context F32 dequant scratch pool for the band kernels (Q1_0, Q4_K/Q5_K/Q6_K, MXFP4, NVFP4).
 *
 * <p>Owned by the matmul context (one per {@link VectorJAM} instance, or one shared by jinfer's
 * legacy tensor path) and passed into each {@code gemm} - NOT a {@code static}/{@code ThreadLocal}.
 * This is the deliberate fix for the old {@code BandGemm.DEQUANT_BAND} ThreadLocal, whose buffers
 * were rooted in the (commonPool, JVM-lifetime) worker threads and so were never released when a
 * context was dropped. Here the buffers are reachable only through this object: reused across every
 * {@code mm} call (no per-call allocation in steady state), and collected with the context that
 * owns the pool.
 *
 * <p>Buffers are NATIVE segments (auto-arena, GC-managed exactly like the float[] they replaced) so
 * the band sweep reads them through the {@link VectorSupport#GLOBAL} pinned-segment route -
 * absolute addresses, bounds/liveness checks folded - the same load path as the activation loads.
 *
 * <p>A worker {@link #acquire}s a buffer at the top of its slice and {@link #release}s it at the
 * end; the pool is a lock-free free-list, so concurrent band workers within one gemm each get their
 * own buffer. The pool retains at most the peak number of concurrent workers' buffers (≤ {@code
 * THREADS}), each grown to the largest {@code k} seen - a few MB at most, freed when the owning
 * context is GC'd.
 */
public final class Scratch {

    private final ConcurrentLinkedQueue<MemorySegment> pool = new ConcurrentLinkedQueue<>();

    /**
     * A native segment of ≥ {@code need} floats (64-byte aligned), reused from the pool when one
     * fits, else freshly allocated on an auto arena (collected with this pool).
     */
    MemorySegment acquire(int need) {
        MemorySegment b;
        while ((b = pool.poll()) != null) {
            if (b.byteSize() >= (long) need * Float.BYTES)
                return b; // discard undersized buffers; a right-sized one replaces them
        }
        return Arena.ofAuto().allocate((long) need * Float.BYTES, 64);
    }

    /** Return a buffer for reuse by a later {@code acquire} (on this or a subsequent gemm). */
    void release(MemorySegment b) {
        pool.offer(b);
    }
}
