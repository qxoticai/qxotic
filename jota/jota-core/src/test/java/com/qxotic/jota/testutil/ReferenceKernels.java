package com.qxotic.jota.testutil;

final class ReferenceKernels {

    private ReferenceKernels() {}

    static float[] gemm(float[] a, float[] b, int m, int n, int k) {
        float[] c = new float[m * n];
        final int tileM = 8;
        final int tileN = 8;
        final int tileK = 8;

        for (int iBase = 0; iBase < m; iBase += tileM) {
            int iLimit = Math.min(iBase + tileM, m);
            for (int jBase = 0; jBase < n; jBase += tileN) {
                int jLimit = Math.min(jBase + tileN, n);
                for (int kBase = 0; kBase < k; kBase += tileK) {
                    int kLimit = Math.min(kBase + tileK, k);
                    for (int i = iBase; i < iLimit; i++) {
                        int aRow = i * k;
                        int cRow = i * n;
                        for (int j = jBase; j < jLimit; j++) {
                            float acc = c[cRow + j];
                            int kk = kBase;
                            for (; kk + 3 < kLimit; kk += 4) {
                                acc += a[aRow + kk] * b[kk * n + j];
                                acc += a[aRow + kk + 1] * b[(kk + 1) * n + j];
                                acc += a[aRow + kk + 2] * b[(kk + 2) * n + j];
                                acc += a[aRow + kk + 3] * b[(kk + 3) * n + j];
                            }
                            for (; kk < kLimit; kk++) {
                                acc += a[aRow + kk] * b[kk * n + j];
                            }
                            c[cRow + j] = acc;
                        }
                    }
                }
            }
        }
        return c;
    }

    static float[] softmaxRows(float[] x, int rows, int cols) {
        float[] y = new float[rows * cols];
        for (int row = 0; row < rows; row++) {
            int base = row * cols;
            float maxVal = x[base];

            int col = 1;
            for (; col + 1 < cols; col += 2) {
                float v0 = x[base + col];
                float v1 = x[base + col + 1];
                if (v0 > maxVal) {
                    maxVal = v0;
                }
                if (v1 > maxVal) {
                    maxVal = v1;
                }
            }
            for (; col < cols; col++) {
                float v = x[base + col];
                if (v > maxVal) {
                    maxVal = v;
                }
            }

            float sumExp = 0.0f;
            col = 0;
            for (; col + 1 < cols; col += 2) {
                float e0 = (float) Math.exp(x[base + col] - maxVal);
                float e1 = (float) Math.exp(x[base + col + 1] - maxVal);
                y[base + col] = e0;
                y[base + col + 1] = e1;
                sumExp += e0 + e1;
            }
            for (; col < cols; col++) {
                float e = (float) Math.exp(x[base + col] - maxVal);
                y[base + col] = e;
                sumExp += e;
            }

            float inv = 1.0f / sumExp;
            col = 0;
            for (; col + 1 < cols; col += 2) {
                y[base + col] *= inv;
                y[base + col + 1] *= inv;
            }
            for (; col < cols; col++) {
                y[base + col] *= inv;
            }
        }
        return y;
    }
}
