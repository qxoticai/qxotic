package com.qxotic.jinfer;

/**
 * The universal floor: a {@code dot()}-based matmul that works for any weight dtype and any operand
 * (including non-F32 activations and the in-place {@code a == c} case). {@code dot()} itself vectorizes
 * when the Vector API is present, so this is "scalar" only in structure (row-at-a-time), not necessarily
 * in execution. It never declines — that is what makes {@link MatMul#mm} a total {@code void} operation:
 * the fast backend ({@link JamMatMul}, over native or Vector API jam) handles what it can, this catches the rest.
 *
 * <p>Relocated from the old base {@code FloatTensor.gemm/gemv} (tiny-serial / in-place-temp / parallel).
 */
final class ScalarMatMul implements MatMul {

    @Override
    public void mm(FloatTensor w, long wOff, int wStride,
                   FloatTensor a, long aOff, int aStride,
                   FloatTensor c, long cOff, int cStride,
                   int m, int n, int k) {
        if (n == 1) {
            if (a != c && (long) m * k <= TINY_MATVEC_ELEMS) {
                // tiny matvec (e.g. the 32-row MoE router): the ForkJoin round trip costs more than the work
                for (int i = 0; i < m; i++) {
                    c.setFloat(cOff + i, w.dot(wOff + (long) i * wStride, a, aOff, k));
                }
            } else if (a == c) {
                // in-place must avoid read-after-write races under parallel execution
                float[] tmp = new float[m];
                Parallel.parallelFor(0, m, i -> tmp[i] = w.dot(wOff + (long) i * wStride, a, aOff, k));
                for (int i = 0; i < m; i++) c.setFloat(cOff + i, tmp[i]);
            } else {
                Parallel.parallelFor(0, m, i -> c.setFloat(cOff + i, w.dot(wOff + (long) i * wStride, a, aOff, k)));
            }
            return;
        }
        // gemm: C[s][row] = dot(W row, A row s)
        if (a == c) {
            float[] tmp = new float[n * m];
            Parallel.parallelFor(0, n * m, idx -> {
                int s = idx / m, row = idx - s * m;
                tmp[idx] = w.dot(wOff + (long) row * wStride, a, aOff + (long) s * aStride, k);
            });
            for (int s = 0; s < n; s++) {
                for (int row = 0; row < m; row++) c.setFloat(cOff + (long) s * cStride + row, tmp[s * m + row]);
            }
        } else {
            Parallel.parallelFor(0, n * m, idx -> {
                int s = idx / m, row = idx - s * m;
                c.setFloat(cOff + (long) s * cStride + row, w.dot(wOff + (long) row * wStride, a, aOff + (long) s * aStride, k));
            });
        }
    }
}
