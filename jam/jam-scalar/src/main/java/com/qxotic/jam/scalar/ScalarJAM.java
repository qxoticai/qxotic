package com.qxotic.jam.scalar;

import com.qxotic.jam.JAM;
import com.qxotic.jam.internal.GGMLType;
import java.lang.foreign.MemorySegment;
import java.lang.ref.Reference;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Pure-Java {@link JAM}: no native code, no Vector API - the portable backend, and the correctness
 * reference every other backend is checked against. Activations and result are F32.
 *
 * <p>Three kernels, all built on the row decoders in {@link Decode}: {@link Gemv} for {@code n ==
 * 1} (one scalar dot per row, memory-bound at any real thread count), {@link Gemm} for batches of
 * {@link Gemm#MIN_N} tokens and up (the token axis as the SIMD axis of an autovectorized tile), and
 * {@link RowGemm} for the batches in between (the same tile over the weight-row axis). The tile
 * runs at roughly half the speed of jam-vector's register tile per core on both C2 and Graal; see
 * {@link Gemm} for the loop shapes the JITs vectorize and the ones they do not.
 *
 * <p>Offsets are BYTE offsets into the operand segments; {@code ldw/lda/ldr} are ELEMENT row
 * strides (the native convention). {@code k} and {@code ldw} must be multiples of the weight's
 * block size ({@link #EINVAL} otherwise). Decodes every jam weight dtype: {@code F32 F16 BF16 Q4_0
 * Q8_0 Q1_0}, the k-quants {@code Q4_K/Q5_K/Q6_K}, and FP4 {@code MXFP4/NVFP4} - the dequant
 * mirrors jam's native reference (jam_ref.h). All parallel work runs on the host's {@link
 * Parallel}; calls are serialized.
 */
public final class ScalarJAM implements JAM {

    private final Parallel parallel;

    /** This instance's buffers; the lock serializes calls because they share them. */
    private final Scratch scratch;

    public ScalarJAM(Parallel parallel) {
        this.parallel = parallel;
        this.scratch = new Scratch(parallel.width());
    }

    private final ReentrantLock lock = new ReentrantLock();

    @Override
    public int mm(
            MemorySegment w,
            long wOff,
            int wt,
            int ldw,
            MemorySegment a,
            long aOff,
            int at,
            int lda,
            MemorySegment r,
            long rOff,
            int rt,
            int ldr,
            int m,
            int n,
            int k) {
        if (at != F32 || rt != F32) return EUNSUPPORTED;
        GGMLType t = GGMLType.byCode(wt);
        if (t == null) return EUNSUPPORTED;
        if (m < 0
                || n < 0
                || k < 0
                || k % t.elementsPerBlock() != 0
                || ldw % t.elementsPerBlock() != 0) return EINVAL;
        if (m == 0 || n == 0) return OK;
        lock.lock();
        try {
            Weight weight = new Weight(w, wOff, t, ldw);
            if (n == 1) Gemv.run(parallel, scratch, weight, a, aOff, r, rOff, m, k);
            else if (n < Gemm.MIN_N && RowGemm.fits(m, n, k, parallel.width()))
                RowGemm.run(parallel, scratch, weight, a, aOff, lda, r, rOff, ldr, m, n, k);
            else Gemm.run(parallel, scratch, weight, a, aOff, lda, r, rOff, ldr, m, n, k);
        } finally {
            lock.unlock();
        }
        // Every read/write goes through the segments' own checked accessors, so liveness is
        // already carried per access; the fences make this backend meet the same explicit contract
        // as NativeJAM/VectorJAM (operands reachable across the whole kernel), belt and braces.
        Reference.reachabilityFence(w);
        Reference.reachabilityFence(a);
        Reference.reachabilityFence(r);
        return OK;
    }
}
