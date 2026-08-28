package com.qxotic.jam.vector;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;

/**
 * Per-context F32 dequant scratch pool for the band kernels (Q1_0, Q4_K/Q5_K/Q6_K, MXFP4, NVFP4).
 *
 * <p>Owned by the matmul context (one per {@link VectorJAM} instance) and passed into each {@code
 * gemm} - NOT a {@code static}/{@code ThreadLocal}. This is the deliberate fix for the old {@code
 * BandGemm.DEQUANT_BAND} ThreadLocal, whose buffers were rooted in the (common pool, JVM-lifetime)
 * worker threads and so were never released when a context was dropped. Here the buffers are
 * reachable only through this object: reused across every {@code mm} call (no per-call allocation
 * in steady state), and collected with the context that owns the pool.
 *
 * <p>Buffers are NATIVE segments (auto-arena, GC-managed exactly like the float[] they replaced) so
 * the band sweep reads them through the {@link VectorSupport#GLOBAL} pinned-segment route -
 * absolute addresses, bounds/liveness checks folded - the same load path as the activation loads.
 *
 * <p>A worker {@link #acquire}s a buffer at the top of its slice and {@link #release}s it at the
 * end; the pool is a lock-free free-list, so concurrent band workers within one gemm each get their
 * own buffer. The pool retains at most the peak number of concurrent workers' buffers (≤ {@code
 * configured parallelism}), each grown to the largest {@code k} seen - a few MB at most, freed when
 * the owning context is GC'd.
 */
public final class Scratch {

    /**
     * Free buffers by size class (byte size rounded up to a 64 KiB multiple). A single mixed queue
     * let a 4 MB packed-activation buffer be taken by a 400 KB panel request and then discarded the
     * small ones on the next big request, so every gemm allocated (and page-faulted) fresh
     * megabytes: measured as run-to-run jitter of 20% and a steady downward drift.
     */
    private final ConcurrentHashMap<Long, ConcurrentLinkedQueue<MemorySegment>> pools =
            new ConcurrentHashMap<>();

    private static final long CLASS = 64 * 1024;

    private static long sizeClass(long bytes) {
        return (bytes + CLASS - 1) / CLASS * CLASS;
    }

    /**
     * Per-worker buffers for the parallel tasks' private panels, indexed by the worker's pool
     * index. A task reuses the buffer its own core wrote last time: buffers rotated through the
     * shared free-list came back from another core's L2, so every dequant store was a remote RFO
     * (measured: L2 misses grew 3.4x from 1 to 8 threads for the same per-thread work).
     */
    private volatile MemorySegment[] perWorker = new MemorySegment[VectorSupport.PARALLELISM + 1];

    /**
     * A 64-byte-aligned buffer of at least {@code need} floats private to the calling worker for
     * the duration of its task. Not released: the slot keeps it for the worker's next task (tasks
     * never nest). Callers outside the pool share one extra slot, serialized by the gemm lock. A
     * ForkJoinPool hands out indices beyond its parallelism (compensating workers), so the slot
     * table grows on demand.
     */
    MemorySegment acquireLocal(long need) {
        int slot = VectorSupport.workerIndex();
        MemorySegment[] slots = perWorker;
        if (slot >= slots.length) {
            synchronized (this) {
                if (slot >= perWorker.length)
                    perWorker =
                            java.util.Arrays.copyOf(
                                    perWorker, Math.max(slot + 1, 2 * perWorker.length));
                slots = perWorker;
            }
        }
        MemorySegment b = slots[slot];
        long bytes = sizeClass(need * Float.BYTES);
        if (b == null || b.byteSize() < bytes) {
            b = Arena.ofAuto().allocate(bytes, 64);
            slots[slot] = b;
        }
        return b;
    }

    /**
     * A 64-byte-aligned buffer of at least {@code need} floats; return it with {@link #release}.
     */
    MemorySegment acquire(long need) {
        long bytes = sizeClass(need * Float.BYTES);
        ConcurrentLinkedQueue<MemorySegment> pool = pools.get(bytes);
        MemorySegment b = pool != null ? pool.poll() : null;
        return b != null ? b : Arena.ofAuto().allocate(bytes, 64);
    }

    void release(MemorySegment b) {
        pools.computeIfAbsent(b.byteSize(), k -> new ConcurrentLinkedQueue<>()).offer(b);
    }
}
