package com.qxotic.jinfer;

import com.qxotic.format.gguf.GGMLType;
import com.qxotic.jam.JAM;
import java.lang.foreign.MemorySegment;

/**
 * Pure-Java {@link JAM} floor: a {@code dot()}-based matmul for ANY weight dtype and any shape —
 * prefill ({@code n>1}) or decode ({@code n==1}). {@code dot()} vectorizes when the Vector API is
 * present, so "scalar" is structural (row-at-a-time), not literal. Mirrors {@link ScalarMatMul} on
 * the JAM segment contract; activations and output are F32 (the JAM contract). It never declines on
 * a supported dtype.
 *
 * <p>Operands are reconstructed as typed {@code FloatTensor}s at their base (offset baked into the
 * slice), so {@code dot()} addresses rows by element stride exactly as the native path does.
 */
final class ScalarJAM implements JAM {

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
        GGMLType t = GGMLType.fromId(wt);
        if (t == null) return EUNSUPPORTED;

        FloatTensor weight = wrap(t, w.asSlice(wOff), (long) m * ldw);
        F32FloatTensor x = new F32FloatTensor((long) n * lda, a.asSlice(aOff));
        F32FloatTensor out = new F32FloatTensor((long) n * ldr, r.asSlice(rOff));

        if (n == 1) {
            Parallel.parallelFor(0, m, i -> out.setFloat(i, weight.dot((long) i * ldw, x, 0L, k)));
        } else {
            Parallel.parallelFor(
                    0,
                    n * m,
                    idx -> {
                        int s = idx / m, row = idx - s * m; // C[s][row] = dot(W row, A row s)
                        out.setFloat(
                                (long) s * ldr + row,
                                weight.dot((long) row * ldw, x, (long) s * lda, k));
                    });
        }
        return OK;
    }

    /** Wrap a native segment as the typed tensor for {@code t} (every JAM weight dtype). */
    static SegmentFloatTensor wrap(GGMLType t, MemorySegment seg, long size) {
        return switch (t) {
            case Q8_0 -> new Q8_0FloatTensor(size, seg);
            case Q4_0 -> new Q4_0FloatTensor(size, seg);
            case Q4_K -> new Q4_KFloatTensor(size, seg);
            case Q5_K -> new Q5_KFloatTensor(size, seg);
            case Q6_K -> new Q6_KFloatTensor(size, seg);
            case MXFP4 -> new MXFP4FloatTensor(size, seg);
            case NVFP4 -> new NVFP4FloatTensor(size, seg);
            case F16 -> new F16FloatTensor(size, seg);
            case BF16 -> new BF16FloatTensor(size, seg);
            case F32 -> new F32FloatTensor(size, seg);
            default -> throw new IllegalArgumentException("ScalarJAM: no tensor for " + t);
        };
    }
}
