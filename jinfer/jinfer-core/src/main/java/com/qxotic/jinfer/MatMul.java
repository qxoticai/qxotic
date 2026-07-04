package com.qxotic.jinfer;

/**
 * One matmul operation, several backends. {@code C = W · Aᵀ} — a gemm when {@code n>1} (prefill), a
 * matvec when {@code n==1} (decode). Row {@code i} of an operand lives at {@code off + i·stride}, in
 * ELEMENTS (the backend converts to bytes via the operand's dtype). Offsets are {@code long} because
 * a weight element index overflows {@code int} on large models; strides and dims are {@code int}.
 *
 * <p>For a quantized weight, {@code wStride} must be block-aligned ({@code wStride % elemsPerBlock == 0}).
 *
 * <p>{@code mm} is total — it always performs the matmul. {@link Dispatch} routes to the fastest
 * applicable backend ({@link JamMatMul}, {@link VectorMatMul}) and falls to the {@link ScalarMatMul}
 * floor, which handles any dtype/operand. There is nothing to signal back, hence {@code void}.
 */
interface MatMul {
    /** Matvec (n==1) at or below this many weight elements stays serial/scalar; above it, the parallel
     *  vector path. The single boundary between ScalarMatMul's tiny path and VectorMatMul's gemv. */
    int TINY_MATVEC_ELEMS = 1 << 18;

    void mm(FloatTensor w, long wOff, int wStride,
            FloatTensor a, long aOff, int aStride,
            FloatTensor c, long cOff, int cStride,
            int m, int n, int k);

    /** The active backend chain. Lives on {@link Dispatch} (a class, run-time-initialized under
     *  native image) - an interface field would be constant-folded at image build time, baking the
     *  builder's jam-less decision into the heap. */
    static MatMul instance() { return Dispatch.ACTIVE; }
}
