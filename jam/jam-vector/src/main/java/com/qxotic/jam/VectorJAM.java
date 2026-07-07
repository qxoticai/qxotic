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

    /** Context-owned dequant scratch for the band kernels — reused across every {@code mm}, GC'd with this
     *  instance (no static/ThreadLocal retention). Assumes single-threaded {@code mm} per instance, like the
     *  other JAM backends; the pool is concurrency-safe across band workers within one call. */
    private final Scratch scratch;

    /** Create the Vector API backend. Throws {@link IllegalStateException} if the {@code jdk.incubator.vector}
     *  module is not on the module path — every kernel needs it. The check runs before any Vector API type is
     *  referenced, so the message is actually reached; a bare kernel reference would instead fail class init
     *  with a cryptic {@code NoClassDefFoundError}. Callers that want a silent fallback probe
     *  {@link #isAvailable()} first (e.g. to select {@code ScalarJAM} instead). */
    public VectorJAM() {
        if (!isAvailable()) {
            throw new IllegalStateException(
                "VectorJAM needs the incubator Vector API module 'jdk.incubator.vector', which is not on the "
              + "module path. Enable it on the java launch (and on javac when compiling): "
              + "--add-modules jdk.incubator.vector");
        }
        this.scratch = new Scratch();
    }

    /** Whether {@code jdk.incubator.vector} is resolved in the boot module layer — i.e. whether
     *  {@code --add-modules jdk.incubator.vector} was in effect. Touches no Vector API type, so it is safe to
     *  call even when the module is absent (unlike constructing the backend or any kernel). */
    public static boolean isAvailable() {
        return ModuleLayer.boot().findModule("jdk.incubator.vector").isPresent();
    }

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
