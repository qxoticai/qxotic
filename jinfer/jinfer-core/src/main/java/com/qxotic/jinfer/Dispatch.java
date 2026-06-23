package com.qxotic.jinfer;

import com.qxotic.format.gguf.GGMLType;

/**
 * Picks a backend per call. The policy is the one we measured: prefill (n&gt;1) is compute-bound so jam
 * wins; decode (n==1) is bandwidth-bound so the Vector backend wins (and jam only when there are no
 * vectors). The scalar backend is the universal floor.
 *
 * <p>Migration state: jam is wired; the Vector/Scalar backends and the decode path are added as dtypes
 * move over. Until then {@code mm} returns false for anything it doesn't yet own, and the caller keeps
 * its existing {@code FloatTensor} path. {@code jamSupports} is the only thing that grows per dtype.
 */
final class Dispatch implements MatMul {

    private final MatMul jam;             // null if jam couldn't load
    private final MatMul vector = new VectorMatMul();   // pure Java; only invoked when USE_VECTOR_API

    private Dispatch(MatMul jam) { this.jam = jam; }

    static Dispatch create() {
        return new Dispatch(JamMatMul.tryLoad() ? new JamMatMul() : null);
    }

    @Override
    public boolean mm(FloatTensor w, long wOff, int wStride,
                      FloatTensor a, long aOff, int aStride,
                      FloatTensor c, long cOff, int cStride,
                      int m, int n, int k) {
        GGMLType t = w.type();
        if (n == 1) {
            // decode: Vector when present (jam can't beat warm in-process vectors); jam when there are
            // none (native 3.7x over scalar); else false -> caller's super.gemv (scalar floor / tiny gemv).
            if (FloatTensor.USE_VECTOR_API) {
                return vector.mm(w, wOff, wStride, a, aOff, aStride, c, cOff, cStride, m, n, k);
            }
            if (jam != null && jamSupports(t, k)) {
                return jam.mm(w, wOff, wStride, a, aOff, aStride, c, cOff, cStride, m, n, k);
            }
            return false;
        }
        // prefill: jam (compute-bound win). Not-yet-migrated / unsupported -> caller's path.
        if (jam != null && jamSupports(t, k)) {
            return jam.mm(w, wOff, wStride, a, aOff, aStride, c, cOff, cStride, m, n, k);
        }
        return false;
    }

    /** dtypes migrated to jam (grows toward the full set, then this method is deleted). jam itself
     *  enforces the exact alignment (e.g. k%256 for K-quants) and declines via false if it's wrong. */
    private static boolean jamSupports(GGMLType t, int k) {
        return switch (t) {
            case Q8_0, Q4_0, Q4_K, Q5_K, Q6_K, MXFP4 -> k % 32 == 0;   // block-quantized: block-aligned k
            case F16, BF16, F32 -> true;                               // dense float: jam handles any k
            default -> false;
        };
    }
}
