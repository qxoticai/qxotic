package com.qxotic.jinfer;

import com.qxotic.format.gguf.GGMLType;

/**
 * Java Vector API backend. Owns the fast vector matvec/gemm. Selected by {@link Dispatch} only when
 * {@code FloatTensor.USE_VECTOR_API} is true, so it never has to re-check that.
 *
 * <p>Migration state: Q8_0 decode (the {@code n==1} matvec) is here — the relocated {@code vectorGemv512}
 * orchestration (4 weight-row streams). The inner vector primitives ({@code gemv512Rows4},
 * {@code vectorDot512F32}) stay on {@code Q8_0FloatTensor} because {@code dot()} shares them and they read
 * its monomorphic segment. Anything not yet moved returns false so the caller keeps its existing path.
 *
 * <p>Offsets are {@code long} end-to-end now (the {@code FloatTensor} accessor API went {@code long}), so
 * there is no narrowing at this boundary — a weight base offset can exceed 2³¹ on large/combined tensors.
 */
final class VectorMatMul implements MatMul {

    @Override
    public boolean mm(FloatTensor w, long wOff, int wStride,
                      FloatTensor a, long aOff, int aStride,
                      FloatTensor c, long cOff, int cStride,
                      int m, int n, int k) {
        if (n != 1 || w.type() != GGMLType.Q8_0) return false;   // only Q8_0 decode migrated so far
        final int bs = GGMLType.Q8_0.getElementsPerBlock();
        // Large + 512-bit + block-aligned -> the streaming vector gemv. Small/unaligned decline to the
        // caller's super.gemv (per-row dot), which wins there (the threshold the old gemv used).
        if (FloatTensor.F_SPECIES.vectorBitSize() != 512 || (long) m * k <= (1 << 18)
                || (k & (bs - 1)) != 0 || (wOff & (bs - 1)) != 0) {
            return false;
        }
        gemvQ8((Q8_0FloatTensor) w, wOff, (F32FloatTensor) a, aOff, (F32FloatTensor) c, cOff, m, k);
        return true;
    }

    /** Relocated vectorGemv512: 4 weight-row streams sharing each activation load; inner kernels on Q8_0FloatTensor. */
    private static void gemvQ8(Q8_0FloatTensor w, long wOff, F32FloatTensor x, long xOff,
                               F32FloatTensor out, long outOff, int m, int k) {
        final int chunk = 64;
        int chunks = (m + chunk - 1) / chunk;
        Parallel.parallelFor(0, chunks, ci -> {
            int rowStart = ci * chunk;
            int rowEnd = Math.min(m, rowStart + chunk);
            int r = rowStart;
            for (; r + 3 < rowEnd; r += 4) {
                Q8_0FloatTensor.gemv512Rows4(w, x, xOff, out, outOff, k, wOff, r);
            }
            for (; r < rowEnd; r++) {
                out.setFloat(outOff + r, Q8_0FloatTensor.vectorDot512F32(w, wOff + (long) r * k, x, xOff, k));
            }
        });
    }
}
