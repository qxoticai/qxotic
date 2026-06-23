package com.qxotic.jinfer;

/**
 * One matmul operation, several backends. {@code C = W · Aᵀ} — a gemm when {@code n>1} (prefill), a
 * matvec when {@code n==1} (decode). Row {@code i} of an operand lives at {@code off + i·stride}, in
 * ELEMENTS (the backend converts to bytes via the operand's dtype). Offsets are {@code long} because
 * a weight element index overflows {@code int} on large models; strides and dims are {@code int}.
 *
 * <p>For a quantized weight, {@code wStride} must be block-aligned ({@code wStride % elemsPerBlock == 0}).
 *
 * <p>During the incremental migration {@code mm} returns {@code false} to let the caller keep its
 * existing path for not-yet-moved dtypes/cases. When every dtype is on this path it becomes total (the
 * scalar backend is the floor) and the boolean goes away.
 */
interface MatMul {
    boolean mm(FloatTensor w, long wOff, int wStride,
               FloatTensor a, long aOff, int aStride,
               FloatTensor c, long cOff, int cStride,
               int m, int n, int k);

    MatMul INSTANCE = Dispatch.create();
}
