package com.qxotic.jinfer;

import com.qxotic.format.gguf.GGMLType;

/**
 * Java Vector API backend for PREFILL: the per-dtype 512-bit register-tiled gemm ({@code n>1}).
 * {@link Dispatch} routes here only when applicable (F32 operands, 512-bit vectors, block-aligned, a
 * tileable dtype), so this never re-checks — it always completes, hence {@code void}.
 *
 * <p>Decode ({@code n==1}) is deliberately NOT here: the scalar floor's parallel one-row {@code dot()}
 * (which vectorizes) measures identical to a specialized Vector gemv on this memory-bound kernel, so there
 * is no separate decode path to maintain — one contiguous stream per row is the whole trick.
 *
 * <p>The tile kernels themselves stay on their tensor classes (they read each tensor's monomorphic
 * segment, shared with {@code dot()}); this just dispatches to them. Offsets are {@code long} end-to-end.
 */
final class VectorMatMul implements MatMul {

    @Override
    public void mm(FloatTensor w, long wOff, int wStride,
                   FloatTensor a, long aOff, int aStride,
                   FloatTensor c, long cOff, int cStride,
                   int m, int n, int k) {
        F32FloatTensor x = (F32FloatTensor) a, out = (F32FloatTensor) c;
        // prefill (aOff == cOff == 0): the register-tiled gemm for this dtype
        switch (w.type()) {
            case Q8_0 -> Q8_0FloatTensor.vectorGemm512F32((Q8_0FloatTensor) w, x, out, aStride, cStride, n, m, k, wOff);
            case Q4_0 -> Q4_0FloatTensor.vectorGemm512((Q4_0FloatTensor) w, x, out, aStride, cStride, n, m, k, wOff);
            case Q4_K -> Q4_KFloatTensor.vectorGemm512((Q4_KFloatTensor) w, x, out, aStride, cStride, n, m, k, wOff);
            case Q5_K -> Q5_KFloatTensor.vectorGemm512((Q5_KFloatTensor) w, x, out, aStride, cStride, n, m, k, wOff);
            case Q6_K -> Q6_KFloatTensor.vectorGemm512((Q6_KFloatTensor) w, x, out, aStride, cStride, n, m, k, wOff);
            case MXFP4 -> MXFP4FloatTensor.vectorGemmMxfp4((MXFP4FloatTensor) w, x, out, aStride, cStride, n, m, k, wOff);
            default -> throw new IllegalStateException("VectorMatMul has no gemm tile for " + w.type());
        }
    }

    /** "vectors present AND 512-bit" — the precondition for every fast path here (constant, JIT-folded). */
    static final boolean IS_512 = FloatTensor.USE_VECTOR_API && FloatTensor.F_SPECIES.vectorBitSize() == 512;

    /** dtypes with a 512-bit prefill tile (the rest fall to the scalar floor). */
    private static boolean hasGemmTile(GGMLType t) {
        return switch (t) {
            case Q8_0, Q4_0, Q4_K, Q5_K, Q6_K, MXFP4 -> true;
            default -> false;   // F16, BF16, F32 -> dot floor
        };
    }

    /** Whether this dtype's 512-bit prefill tile applies (block-aligned k and weight offset). */
    static boolean gemmApplies(GGMLType t, int k, long wOff) {
        if (!IS_512 || !hasGemmTile(t)) return false;
        int blk = t.getElementsPerBlock();
        return (k % blk == 0) && (wOff % blk == 0);
    }
}
