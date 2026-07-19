package com.qxotic.jinfer;

import com.qxotic.format.gguf.GGMLType;
import com.qxotic.jam.JAM;

/**
 * Runs any {@link JAM} backend (native {@code NativeJAM} or Vector API {@code VectorJAM}) over
 * jinfer's {@link SegmentFloatTensor} view. This is the one place the tensor-to-JAM translation
 * lives: an operand's {@code (vseg, vbase)} = (GLOBAL segment, absolute byte base) with ELEMENT
 * offsets/strides becomes JAM's {@code (segment, BYTE operand offset, element stride)} contract, so
 * jam reads it zero-copy and bounds-checks against {@code vseg}. On a runtime decline ({@code st !=
 * OK} — EBUSY contention, or a shape the backend won't take) it hands off to the scalar floor.
 * {@link Dispatch} gates every call, so a decline is rare. Offsets are {@code long} (a weight byte
 * offset can exceed 2³¹ on large models).
 */
final class JamMatMul implements MatMul {

    private final JAM jam;
    private final MatMul fallback; // scalar floor, for a runtime decline

    JamMatMul(JAM jam, MatMul fallback) {
        this.jam = jam;
        this.fallback = fallback;
    }

    @Override
    public void mm(
            FloatTensor w,
            long wOff,
            int wStride,
            FloatTensor a,
            long aOff,
            int aStride,
            FloatTensor c,
            long cOff,
            int cStride,
            int m,
            int n,
            int k) {
        GGMLType t = w.type();
        SegmentFloatTensor sw = (SegmentFloatTensor) w,
                sa = (SegmentFloatTensor) a,
                sc = (SegmentFloatTensor) c;
        // wOff/aOff/cOff are ELEMENT offsets (Dispatch guarantees wOff is block-aligned, so the
        // block-aware
        // weight byte offset is exact); F32 activation/result are 4 bytes/element.
        long wByte = sw.vbase + (wOff / t.getElementsPerBlock()) * t.getBlockByteSize();
        long aByte = sa.vbase + aOff * Float.BYTES;
        long cByte = sc.vbase + cOff * Float.BYTES;
        int st =
                jam.mm(
                        sw.vseg, wByte, jamTag(t), wStride, sa.vseg, aByte, JAM.F32, aStride,
                        sc.vseg, cByte, JAM.F32, cStride, m, n, k);
        if (st != JAM.OK) // EBUSY (concurrent same-context) / unsupported shape -> the floor
            // finishes it
            fallback.mm(w, wOff, wStride, a, aOff, aStride, c, cOff, cStride, m, n, k);
    }

    /** GGMLType -> jam dtype tag (== ggml_type value, but mapped explicitly to stay honest). */
    static int jamTag(GGMLType t) {
        return switch (t) {
            case Q8_0 -> JAM.Q8_0;
            case Q4_0 -> JAM.Q4_0;
            case Q4_K -> JAM.Q4_K;
            case Q5_K -> JAM.Q5_K;
            case Q6_K -> JAM.Q6_K;
            case MXFP4 -> JAM.MXFP4;
            case NVFP4 -> JAM.NVFP4;
            case Q1_0 -> JAM.Q1_0;
            case F16 -> JAM.F16;
            case BF16 -> JAM.BF16;
            case F32 -> JAM.F32;
            default -> throw new IllegalArgumentException("jam has no kernel for " + t);
        };
    }
}
