package com.qxotic.jam;

import org.junit.jupiter.api.Test;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests the jam Java API end to end: the native library loads (via NativeLoader), and JAM.mm computes
 * C = W @ Aᵀ correctly for F32 and Q8_0 weights, and reports the right status for bad/unsupported calls.
 * Operands are off-heap (FFM). The native lib is supplied by -Djam.library.path (see pom.xml).
 */
class JamTest {

    /** F32 matmul through the API; returns C [m×n] row-major. */
    private static float[] mmF32(Arena ar, float[] W, float[] A, int m, int n, int k) {
        MemorySegment w = ar.allocate((long) m * k * Float.BYTES);
        MemorySegment a = ar.allocate((long) n * k * Float.BYTES);
        MemorySegment c = ar.allocate((long) m * n * Float.BYTES);
        MemorySegment.copy(W, 0, w, JAVA_FLOAT, 0, m * k);
        MemorySegment.copy(A, 0, a, JAVA_FLOAT, 0, n * k);
        int st = JAM.mm(w, 0, JAM.F32, k, a, 0, JAM.F32, k, c, 0, JAM.F32, m, m, n, k);   // safe MemorySegment API
        assertEquals(JAM.OK, st, "jam_mm should return OK");
        float[] out = new float[m * n];
        MemorySegment.copy(c, JAVA_FLOAT, 0, out, 0, m * n);   // token-major: out[j*m + i]
        return out;
    }

    private static float[] refMM(float[] W, float[] A, int m, int n, int k) {
        float[] r = new float[m * n];                          // token-major to match jam: r[j*m + i]
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++) {
                double s = 0;
                for (int t = 0; t < k; t++) s += (double) W[i * k + t] * A[j * k + t];
                r[j * m + i] = (float) s;
            }
        return r;
    }

    @Test
    void nativeLibraryLoadsAndComputes() {           // touching JAM triggers NativeLoader.load()
        try (Arena ar = Arena.ofConfined()) {
            float[] c = mmF32(ar, new float[]{1, 2, 3, 4}, new float[]{5, 6, 7, 8}, 1, 1, 4);
            assertEquals(70.0f, c[0], 1e-4f);        // dot([1,2,3,4],[5,6,7,8]) = 5+12+21+32
        }
    }

    @Test
    void f32Matmul() {
        try (Arena ar = Arena.ofConfined()) {
            int m = 5, n = 3, k = 17;                // non-aligned shape (mnpack edges)
            float[] W = new float[m * k], A = new float[n * k];
            for (int i = 0; i < W.length; i++) W[i] = (float) Math.sin(i * 0.3);
            for (int i = 0; i < A.length; i++) A[i] = (float) Math.cos(i * 0.2);
            assertArrayEquals(refMM(W, A, m, n, k), mmF32(ar, W, A, m, n, k), 1e-3f);
        }
    }

    @Test
    void gemv() {                                    // n == 1, the decode case
        try (Arena ar = Arena.ofConfined()) {
            int m = 8, k = 32;
            float[] W = new float[m * k], A = new float[k];
            for (int i = 0; i < W.length; i++) W[i] = (i % 7) - 3;
            for (int i = 0; i < k; i++) A[i] = (i % 5) - 2;
            assertArrayEquals(refMM(W, A, m, 1, k), mmF32(ar, W, A, m, 1, k), 1e-3f);
        }
    }

    @Test
    void q8_0Matmul() {                              // quantize the weight to Q8_0 in Java, run, compare
        try (Arena ar = Arena.ofConfined()) {
            int m = 4, n = 3, k = 64;                // 2 blocks of 32
            int nb = k / 32;
            float[] Wf = new float[m * k], A = new float[n * k];
            for (int i = 0; i < Wf.length; i++) Wf[i] = (float) Math.sin(i * 0.11);
            for (int i = 0; i < A.length; i++) A[i] = (float) Math.cos(i * 0.07);

            MemorySegment w = ar.allocate((long) m * nb * 34);   // block_q8_0 = { fp16 d; int8 qs[32] }
            float[] wdq = new float[m * k];                      // dequantized weight (kernel's view)
            for (int i = 0; i < m; i++)
                for (int b = 0; b < nb; b++) {
                    float amax = 0;
                    for (int t = 0; t < 32; t++) amax = Math.max(amax, Math.abs(Wf[i * k + b * 32 + t]));
                    float d = amax / 127f, id = d > 0 ? 1 / d : 0;
                    float dq = Float.float16ToFloat(Float.floatToFloat16(d));   // fp16 round-trip
                    long base = (long) (i * nb + b) * 34;
                    w.set(JAVA_SHORT, base, Float.floatToFloat16(d));
                    for (int t = 0; t < 32; t++) {
                        int q = Math.max(-128, Math.min(127, Math.round(Wf[i * k + b * 32 + t] * id)));
                        w.set(JAVA_BYTE, base + 2 + t, (byte) q);
                        wdq[i * k + b * 32 + t] = q * dq;
                    }
                }
            MemorySegment a = ar.allocate((long) n * k * Float.BYTES);
            MemorySegment c = ar.allocate((long) m * n * Float.BYTES);
            MemorySegment.copy(A, 0, a, JAVA_FLOAT, 0, n * k);
            int st = JAM.mm(w, 0, JAM.Q8_0, k, a, 0, JAM.F32, k, c, 0, JAM.F32, m, m, n, k);   // safe MemorySegment API
            assertEquals(JAM.OK, st);

            float[] ref = refMM(wdq, A, m, n, k);    // token-major; kernel may requant A -> loose tol
            for (int i = 0; i < m * n; i++)
                assertEquals(ref[i], c.getAtIndex(JAVA_FLOAT, i), 0.05f + 0.05f * Math.abs(ref[i]),
                        "Q8_0 result within int8 activation-requant tolerance");
        }
    }

    @Test
    void f16WeightsSupported() {
        try (Arena ar = Arena.ofConfined()) {
            MemorySegment w = ar.allocate(64), a = ar.allocate(64), c = ar.allocate(64);
            int st = JAM.mmUnsafe(w.address(), JAM.F16, 16, a.address(), JAM.F32, 16, c.address(), JAM.F32, 1, 1, 1, 16);
            assertEquals(JAM.OK, st, "F16 dense weights are supported (AVX-512 tile, else the generic float floor)");
        }
    }

    @Test
    void invalidArgsReturnEinval() {
        try (Arena ar = Arena.ofConfined()) {
            MemorySegment w = ar.allocate(64), a = ar.allocate(64), c = ar.allocate(64);
            assertEquals(JAM.EINVAL,
                    JAM.mmUnsafe(w.address(), JAM.F32, 4, a.address(), JAM.F32, 4, c.address(), JAM.F32, 1, 0, 1, 4),
                    "m <= 0");
            assertEquals(JAM.EINVAL,
                    JAM.mmUnsafe(w.address(), JAM.F32, 2, a.address(), JAM.F32, 4, c.address(), JAM.F32, 1, 1, 1, 4),
                    "ldw < k");
            assertEquals(JAM.EINVAL,
                    JAM.mmUnsafe(0L, JAM.F32, 4, a.address(), JAM.F32, 4, c.address(), JAM.F32, 1, 1, 1, 4),
                    "null weight pointer");
        }
    }
}
