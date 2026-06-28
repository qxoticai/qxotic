package com.qxotic.jam;

import java.lang.foreign.MemorySegment;

/**
 * Java Vector API {@link JAM} backend — jam-vector's self-contained matmul, peer to {@link ScalarJAM} and
 * {@code NativeJAM}. Each tileable weight dtype is dispatched straight to its register-tiled (Q8_0/Q4_0) or
 * dequant-to-scratch band (k-quant/FP4) kernel on the raw operand segments — no tensor reconstruction.
 *
 * <p>PREFILL only ({@code n > 1}) with F32 activations + result. Decode ({@code n == 1}, bandwidth-bound),
 * non-tileable dtypes, strided weights, and the absence of a usable SIMD width all return
 * {@link #EUNSUPPORTED} for the caller's scalar floor. Activation/output are addressed through
 * {@link VectorSupport#GLOBAL} at absolute addresses (one segment type, so the access folds), exactly as
 * the kernels' own tests drive them.
 *
 * <p>The register-tile shape and vector width are resolved once in {@link VectorSupport} — auto by default,
 * overridable with {@code -Djam.vector.tile} / {@code -Djam.vector.width}; the k-quant/FP4 decoders run at
 * 128/256/512-bit, so this backend is no longer pinned to AVX-512.
 */
public final class VectorJAM implements JAM {

    /** Whether a 512-bit float species is in play (informational; the kernels run at 128/256/512). */
    public static final boolean IS_512 = VectorSupport.IS_512;

    /** Context-owned dequant scratch for the band kernels — reused across every {@code mm}, GC'd with this
     *  instance (no static/ThreadLocal retention). Assumes single-threaded {@code mm} per instance, like the
     *  other JAM backends; the pool is concurrency-safe across band workers within one call. */
    private final Scratch scratch = new Scratch();

    private static boolean tileable(int wt) {
        return wt == Q8_0 || wt == Q4_0 || wt == Q4_K || wt == Q5_K || wt == Q6_K || wt == MXFP4 || wt == NVFP4;
    }

    @Override
    public int mm(MemorySegment w, long wOff, int wt, int ldw,
                  MemorySegment a, long aOff, int at, int lda,
                  MemorySegment r, long rOff, int rt, int ldr,
                  int m, int n, int k) {
        if (n <= 1 || at != F32 || rt != F32 || VectorSupport.F_SPECIES.vectorBitSize() < 128) return EUNSUPPORTED;
        GGMLType t = GGMLType.byCode(wt);
        if (t == null || !tileable(wt)) return EUNSUPPORTED;
        if (ldw != k || k % t.blockElems != 0) return EUNSUPPORTED;   // contiguous weight rows, whole blocks

        // Weight read relative to its slice (kernel wOff = 0); activation/output via GLOBAL at absolute base.
        MemorySegment ws = w.asSlice(wOff);
        MemorySegment g = VectorSupport.GLOBAL;
        long ab = a.address() + aOff;
        long ob = r.address() + rOff;
        switch (wt) {
            case Q8_0  -> Q8Kernel.gemm(ws, g, ab, g, ob, lda, ldr, n, m, k, 0L);
            case Q4_0  -> Q4Kernel.gemm(ws, g, ab, g, ob, lda, ldr, n, m, k, 0L);
            case Q4_K  -> Q4KKernel.gemm(ws, g, ab, g, ob, lda, ldr, n, m, k, 0L, scratch);
            case Q5_K  -> Q5KKernel.gemm(ws, g, ab, g, ob, lda, ldr, n, m, k, 0L, scratch);
            case Q6_K  -> Q6KKernel.gemm(ws, g, ab, g, ob, lda, ldr, n, m, k, 0L, scratch);
            case MXFP4 -> Mxfp4Kernel.gemm(ws, g, ab, g, ob, lda, ldr, n, m, k, 0L, scratch);
            case NVFP4 -> Nvfp4Kernel.gemm(ws, g, ab, g, ob, lda, ldr, n, m, k, 0L, scratch);
            default -> { return EUNSUPPORTED; }
        }
        return OK;
    }
}
