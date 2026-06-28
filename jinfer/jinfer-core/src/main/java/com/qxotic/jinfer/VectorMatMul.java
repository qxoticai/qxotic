package com.qxotic.jinfer;

import com.qxotic.format.gguf.GGMLType;
import com.qxotic.jam.JAM;
import com.qxotic.jam.VectorJAM;

/**
 * Java Vector API backend for PREFILL ({@code n>1}): the per-dtype register-tiled / dequant-band gemm.
 * {@link Dispatch} routes here only when applicable (F32 operands, 512-bit vectors, block-aligned, a
 * tileable dtype) per {@link #gemmApplies}.
 *
 * <p>This is now a thin adapter onto {@link VectorJAM} (jam-vector), which owns the relocated kernels and
 * the dequant scratch. The only work here is translating jinfer's tensor view — operands carried as
 * {@code (vseg, vbase)} = (GLOBAL segment, absolute byte base) with ELEMENT offsets/strides — into the
 * {@link JAM} segment contract (byte operand offsets, element strides). VectorJAM declines a handful of
 * shapes it can't tile (e.g. a strided weight); those fall to the scalar floor, keeping {@code mm} total.
 *
 * <p>Decode ({@code n==1}) is deliberately NOT here: the scalar floor's parallel one-row {@code dot()}
 * measures identical to a specialized Vector gemv on that memory-bound kernel.
 */
final class VectorMatMul implements MatMul {

    private final JAM vector = new VectorJAM();   // owns the kernels + its per-context dequant scratch
    private final MatMul fallback;                // scalar floor, for the rare shape VectorJAM declines

    VectorMatMul(MatMul fallback) {
        this.fallback = fallback;
    }

    @Override
    public void mm(FloatTensor w, long wOff, int wStride,
                   FloatTensor a, long aOff, int aStride,
                   FloatTensor c, long cOff, int cStride,
                   int m, int n, int k) {
        SegmentFloatTensor sw = (SegmentFloatTensor) w, x = (SegmentFloatTensor) a, out = (SegmentFloatTensor) c;
        GGMLType t = w.type();
        // (vseg, vbase) is (GLOBAL, absolute byte base); JAM takes byte operand offsets + element strides.
        // wOff/aOff/cOff are ELEMENT offsets (gemmApplies guarantees wOff is block-aligned, so the weight
        // byte offset is exact); F32 activation/result are 4 bytes/element.
        long wByte = sw.vbase + (wOff / t.getElementsPerBlock()) * (long) t.getBlockByteSize();
        long aByte = x.vbase + aOff * (long) Float.BYTES;
        long cByte = out.vbase + cOff * (long) Float.BYTES;
        int st = vector.mm(sw.vseg, wByte, t.getId(), wStride,
                           x.vseg, aByte, JAM.F32, aStride,
                           out.vseg, cByte, JAM.F32, cStride, m, n, k);
        if (st != JAM.OK)
            fallback.mm(w, wOff, wStride, a, aOff, aStride, c, cOff, cStride, m, n, k);
    }

    /** "vectors present AND 512-bit" — the precondition for every fast path here (constant, JIT-folded). */
    static final boolean IS_512 = FloatTensor.USE_VECTOR_API && FloatTensor.F_SPECIES.vectorBitSize() == 512;

    /** dtypes with a register-tiled prefill kernel (the rest fall to the scalar floor). */
    private static boolean hasGemmTile(GGMLType t) {
        return switch (t) {
            case Q8_0, Q4_0, Q4_K, Q5_K, Q6_K, MXFP4, NVFP4 -> true;
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
