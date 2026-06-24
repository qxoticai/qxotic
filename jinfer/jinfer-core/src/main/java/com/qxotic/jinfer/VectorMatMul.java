package com.qxotic.jinfer;

import com.qxotic.format.gguf.GGMLType;

/**
 * Java Vector API backend. Owns the fast register-tiled paths: the Q8_0 streaming matvec (decode) and the
 * per-dtype tiled gemm (prefill, for the dtypes that have a 512-bit tile). {@link Dispatch} only routes
 * here when the call is applicable (F32 operands, 512-bit vectors, block-aligned, a tileable dtype), so
 * this never has to re-check — it always completes, hence {@code void}.
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
        if (n == 1) {                       // decode: only Q8_0 has a dedicated streaming gemv
            gemvQ8((Q8_0FloatTensor) w, wOff, x, aOff, out, cOff, m, k);
            return;
        }
        // prefill (aOff == cOff == 0): the register-tiled gemm for this dtype
        switch (w.type()) {
            case Q8_0 -> Q8_0FloatTensor.vectorGemm512F32((Q8_0FloatTensor) w, x, out, aStride, cStride, n, m, k, wOff);
            case Q4_0 -> Q4_0FloatTensor.vectorGemm512((Q4_0FloatTensor) w, x, out, aStride, cStride, n, m, k, wOff);
            case Q4_K -> Q4_KFloatTensor.vectorGemm512((Q4_KFloatTensor) w, x, out, aStride, cStride, n, m, k, wOff);
            case Q6_K -> Q6_KFloatTensor.vectorGemm512((Q6_KFloatTensor) w, x, out, aStride, cStride, n, m, k, wOff);
            case MXFP4 -> MXFP4FloatTensor.vectorGemmMxfp4((MXFP4FloatTensor) w, x, out, aStride, cStride, n, m, k, wOff);
            default -> throw new IllegalStateException("VectorMatMul has no gemm tile for " + w.type());
        }
    }

    /**
     * Q8_0 streaming matvec. Each thread walks its 64-row band ONE row fully at a time: because the weight
     * rows are contiguous, a band is a single sequential memory stream the HW prefetcher rides cleanly.
     * (Interleaving 4 rows per step — sharing the activation load — saves compute but splits the band into
     * 4 strided streams; on this memory-bound kernel that costs ~3% decode, so we don't.)
     */
    private static void gemvQ8(Q8_0FloatTensor w, long wOff, F32FloatTensor x, long xOff,
                               F32FloatTensor out, long outOff, int m, int k) {
        final int chunk = 64;
        int chunks = (m + chunk - 1) / chunk;
        Parallel.parallelFor(0, chunks, ci -> {
            int rowStart = ci * chunk;
            int rowEnd = Math.min(m, rowStart + chunk);
            for (int r = rowStart; r < rowEnd; r++) {
                out.setFloat(outOff + r, Q8_0FloatTensor.vectorDot512F32(w, wOff + (long) r * k, x, xOff, k));
            }
        });
    }

    /** "vectors present AND 512-bit" — the precondition for every fast path here (constant, JIT-folded). */
    static final boolean IS_512 = FloatTensor.USE_VECTOR_API && FloatTensor.F_SPECIES.vectorBitSize() == 512;

    /** dtypes with a 512-bit prefill tile (the rest fall to the scalar floor). */
    private static boolean hasGemmTile(GGMLType t) {
        return switch (t) {
            case Q8_0, Q4_0, Q4_K, Q6_K, MXFP4 -> true;
            default -> false;   // Q5_K, F16, BF16, F32 -> dot floor
        };
    }

    /** Whether the Q8_0 streaming matvec applies (else the dot floor handles the matvec). */
    static boolean gemvApplies(GGMLType t, int m, int k, long wOff) {
        int bs = GGMLType.Q8_0.getElementsPerBlock();
        return t == GGMLType.Q8_0 && IS_512
                && (long) m * k > TINY_MATVEC_ELEMS
                && (k & (bs - 1)) == 0 && (wOff & (bs - 1)) == 0;
    }

    /** Whether this dtype's 512-bit prefill tile applies (block-aligned k and weight offset). */
    static boolean gemmApplies(GGMLType t, int k, long wOff) {
        if (!IS_512 || !hasGemmTile(t)) return false;
        int blk = t.getElementsPerBlock();
        return (k % blk == 0) && (wOff % blk == 0);
    }
}
