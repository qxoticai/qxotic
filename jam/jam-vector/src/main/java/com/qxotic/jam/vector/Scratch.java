package com.qxotic.jam.vector;

import com.qxotic.jam.JAM;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

/**
 * A {@link VectorJAM} instance's context: the host pool its regions run on and the F32 scratch its
 * kernels reuse across calls - one packed-activation buffer per gemm and one dequant panel per slot
 * of the pool. Passed into every kernel; never a {@code static} or a {@code ThreadLocal}, so the
 * buffers are released with the instance (the old {@code BandGemm.DEQUANT_BAND} ThreadLocal was
 * rooted in worker threads and outlived every context).
 *
 * <p>Buffers are native segments (auto-arena) so the sweeps read them through {@link
 * VectorSupport#GLOBAL} at absolute addresses, the same load path as the activations. A slot's
 * panel is reused by whichever task runs on that slot next, so every dequant store lands in the L2
 * of the core that wrote it last time (rotating buffers across cores measured 3.4x more L2 misses
 * at 8 threads).
 */
public final class Scratch {

    private final JAM.Parallel parallel;
    private final MemorySegment[] perSlot;
    private MemorySegment packed = MemorySegment.NULL;

    public Scratch(JAM.Parallel parallel) {
        this.parallel = parallel;
        this.perSlot = new MemorySegment[parallel.width()];
    }

    /** The pool every region of this context runs on. */
    JAM.Parallel parallel() {
        return parallel;
    }

    /** {@code parallel().width()}: the slots, and the most tasks a region runs at once. */
    int width() {
        return parallel.width();
    }

    private static final long CLASS = 64 * 1024;

    private static MemorySegment grown(MemorySegment b, long floats) {
        long bytes = (floats * Float.BYTES + CLASS - 1) / CLASS * CLASS;
        return b.byteSize() >= bytes ? b : Arena.ofAuto().allocate(bytes, 64);
    }

    /** The 64-byte-aligned buffer of at least {@code need} floats owned by {@code slot}. */
    MemorySegment local(int slot, long need) {
        MemorySegment b = perSlot[slot];
        if (b == null || b.byteSize() < need * Float.BYTES)
            perSlot[slot] = b = grown(b == null ? MemorySegment.NULL : b, need);
        return b;
    }

    /**
     * The 64-byte-aligned packed-activation buffer of at least {@code need} floats; one per call.
     */
    MemorySegment packed(long need) {
        return packed = grown(packed, need);
    }
}
