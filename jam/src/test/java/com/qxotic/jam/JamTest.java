package com.qxotic.jam;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.function.Executable;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;

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

    @Test
    void unsupportedCombosReturnEunsupported() {        // dispatch can't serve these -> EUNSUPPORTED (≠ EINVAL)
        try (Arena ar = Arena.ofConfined()) {
            MemorySegment w = ar.allocate(8192), a = ar.allocate(8192), c = ar.allocate(8192);
            int m = 2, n = 2, k = 64;
            assertEquals(JAM.EUNSUPPORTED,
                    JAM.mmUnsafe(w.address(), JAM.Q2_K, k, a.address(), JAM.F32, k, c.address(), JAM.F32, m, m, n, k),
                    "unimplemented weight dtype");
            assertEquals(JAM.EUNSUPPORTED,
                    JAM.mmUnsafe(w.address(), JAM.F32, k, a.address(), JAM.F16, k, c.address(), JAM.F32, m, m, n, k),
                    "non-F32 activation");
            assertEquals(JAM.EUNSUPPORTED,
                    JAM.mmUnsafe(w.address(), JAM.F32, k, a.address(), JAM.F32, k, c.address(), JAM.F16, m, m, n, k),
                    "non-F32 output");
            assertEquals(JAM.EUNSUPPORTED,
                    JAM.mmUnsafe(w.address(), JAM.Q8_0, 48, a.address(), JAM.F32, 48, c.address(), JAM.F32, m, m, n, 48),
                    "k not a multiple of the Q8_0 block (32)");
        }
    }

    /** assert the bounds-checked mm() rejects this call in JAVA (IndexOutOfBoundsException), never reaching native. */
    private static void rejects(String why, Executable mmCall) {
        assertThrows(IndexOutOfBoundsException.class, mmCall, why);
    }

    @Test
    void boundsRejectUndersizedOperands() {             // each operand one byte short -> throws; exact-fit -> OK
        try (Arena ar = Arena.ofConfined()) {
            int m = 4, n = 3, k = 64;
            long wB = (long) m * k * 4, aB = (long) n * k * 4, cB = (long) m * n * 4;
            MemorySegment w = ar.allocate(wB), a = ar.allocate(aB), c = ar.allocate(cB);
            assertEquals(JAM.OK, JAM.mm(w, 0, JAM.F32, k, a, 0, JAM.F32, k, c, 0, JAM.F32, m, m, n, k), "exact fit, no false positive");
            rejects("weight -1B",     () -> JAM.mm(ar.allocate(wB - 1), 0, JAM.F32, k, a, 0, JAM.F32, k, c, 0, JAM.F32, m, m, n, k));
            rejects("activation -1B", () -> JAM.mm(w, 0, JAM.F32, k, ar.allocate(aB - 1), 0, JAM.F32, k, c, 0, JAM.F32, m, m, n, k));
            rejects("output -1B",     () -> JAM.mm(w, 0, JAM.F32, k, a, 0, JAM.F32, k, ar.allocate(cB - 1), 0, JAM.F32, m, m, n, k));
        }
    }

    @Test
    void boundsRejectBadOffsets() {                      // a byte offset must be counted + must be non-negative
        try (Arena ar = Arena.ofConfined()) {
            int m = 4, n = 3, k = 64;
            MemorySegment w = ar.allocate((long) m * k * 4), a = ar.allocate((long) n * k * 4), c = ar.allocate((long) m * n * 4);
            rejects("weight offset overflow", () -> JAM.mm(w, 4, JAM.F32, k, a, 0, JAM.F32, k, c, 0, JAM.F32, m, m, n, k));
            rejects("act offset overflow",    () -> JAM.mm(w, 0, JAM.F32, k, a, 4, JAM.F32, k, c, 0, JAM.F32, m, m, n, k));
            rejects("out offset overflow",    () -> JAM.mm(w, 0, JAM.F32, k, a, 0, JAM.F32, k, c, 4, JAM.F32, m, m, n, k));
            rejects("negative offset",        () -> JAM.mm(w, -4, JAM.F32, k, a, 0, JAM.F32, k, c, 0, JAM.F32, m, m, n, k));
        }
    }

    @Test
    void boundsHonorStrides() {                          // lda/ldb/ldc are ELEMENT strides; size = (rows-1)*stride + lastRow
        try (Arena ar = Arena.ofConfined()) {
            int m = 4, n = 3, k = 64, pad = 8;
            MemorySegment w = ar.allocate((long) m * k * 4), a = ar.allocate((long) n * k * 4), c = ar.allocate((long) m * n * 4);
            // strided weight (lda = k+pad)
            int lda = k + pad; long wN = (long) (m - 1) * lda * 4 + (long) k * 4;
            assertEquals(JAM.OK, JAM.mm(ar.allocate(wN), 0, JAM.F32, lda, a, 0, JAM.F32, k, c, 0, JAM.F32, m, m, n, k), "strided weight exact");
            rejects("strided weight -1B", () -> JAM.mm(ar.allocate(wN - 1), 0, JAM.F32, lda, a, 0, JAM.F32, k, c, 0, JAM.F32, m, m, n, k));
            // strided activation (ldb = k+pad)
            int ldb = k + pad; long aN = (long) (n - 1) * ldb * 4 + (long) k * 4;
            assertEquals(JAM.OK, JAM.mm(w, 0, JAM.F32, k, ar.allocate(aN), 0, JAM.F32, ldb, c, 0, JAM.F32, m, m, n, k), "strided act exact");
            rejects("strided act -1B", () -> JAM.mm(w, 0, JAM.F32, k, ar.allocate(aN - 1), 0, JAM.F32, ldb, c, 0, JAM.F32, m, m, n, k));
            // padded output (ldc = m+pad) — the token-major C stride
            int ldc = m + pad; long cN = (long) (n - 1) * ldc * 4 + (long) m * 4;
            assertEquals(JAM.OK, JAM.mm(w, 0, JAM.F32, k, a, 0, JAM.F32, k, ar.allocate(cN), 0, JAM.F32, ldc, m, n, k), "padded output exact");
            rejects("padded output -1B", () -> JAM.mm(w, 0, JAM.F32, k, a, 0, JAM.F32, k, ar.allocate(cN - 1), 0, JAM.F32, ldc, m, n, k));
        }
    }

    @Test
    void boundsBlockAwareAndDense() {                    // weight bytes are block-aware (quant) / 2-per-elem (F16/BF16)
        try (Arena ar = Arena.ofConfined()) {
            int m = 4, n = 3, k = 256;                   // k%256 so Q4_K/Q5_K/Q6_K are legal
            MemorySegment a = ar.allocate((long) n * k * 4), c = ar.allocate((long) m * n * 4);
            long q8 = (long) m * (k / 32) * 34, q4k = (long) m * (k / 256) * 144, half = (long) m * k * 2;
            assertEquals(JAM.OK, JAM.mm(ar.allocate(q8), 0, JAM.Q8_0, k, a, 0, JAM.F32, k, c, 0, JAM.F32, m, m, n, k), "Q8_0 exact");
            rejects("Q8_0 -1B", () -> JAM.mm(ar.allocate(q8 - 1), 0, JAM.Q8_0, k, a, 0, JAM.F32, k, c, 0, JAM.F32, m, m, n, k));
            rejects("Q4_K -1B", () -> JAM.mm(ar.allocate(q4k - 1), 0, JAM.Q4_K, k, a, 0, JAM.F32, k, c, 0, JAM.F32, m, m, n, k));
            rejects("F16 -1B",  () -> JAM.mm(ar.allocate(half - 1), 0, JAM.F16, k, a, 0, JAM.F32, k, c, 0, JAM.F32, m, m, n, k));
            rejects("BF16 -1B", () -> JAM.mm(ar.allocate(half - 1), 0, JAM.BF16, k, a, 0, JAM.F32, k, c, 0, JAM.F32, m, m, n, k));
        }
    }

    @Test
    void boundsSkipWhenNativeClassifies() {              // must NOT throw where native cleanly returns a status
        try (Arena ar = Arena.ofConfined()) {
            int m = 2, n = 2, k = 64;
            MemorySegment w = ar.allocate((long) m * k * 4), a = ar.allocate((long) n * k * 4), c = ar.allocate((long) m * n * 4);
            assertEquals(JAM.EUNSUPPORTED, JAM.mm(w, 0, JAM.Q2_K, k, a, 0, JAM.F32, k, c, 0, JAM.F32, m, m, n, k), "unsupported weight dtype");
            assertEquals(JAM.EUNSUPPORTED, JAM.mm(w, 0, JAM.F32, k, a, 0, JAM.F16, k, c, 0, JAM.F32, m, m, n, k), "non-F32 activation");
            assertEquals(JAM.EUNSUPPORTED, JAM.mm(w, 0, JAM.F32, k, a, 0, JAM.F32, k, c, 0, JAM.F16, m, m, n, k), "non-F32 output");
            assertEquals(JAM.EINVAL,       JAM.mm(w, 0, JAM.F32, k, a, 0, JAM.F32, k, c, 0, JAM.F32, m, 0, n, k), "m<=0 (guard skips bounds)");
            assertEquals(JAM.EINVAL,       JAM.mm(w, 0, JAM.F32, k - 1, a, 0, JAM.F32, k, c, 0, JAM.F32, m, m, n, k), "ldw<k (guard skips bounds)");
        }
    }

    @Test
    void nonZeroOffsetMatmul() {                        // sub-segment placement: address()+off arithmetic + bounds
        try (Arena ar = Arena.ofConfined()) {
            int m = 3, n = 2, k = 32;
            long wOff = 64, aOff = 32, cOff = 16;       // byte offsets, all 4-aligned
            float[] W = new float[m * k], A = new float[n * k];
            for (int i = 0; i < W.length; i++) W[i] = (float) Math.sin(i * 0.21);
            for (int i = 0; i < A.length; i++) A[i] = (float) Math.cos(i * 0.13);
            MemorySegment w = ar.allocate(wOff + (long) m * k * 4);
            MemorySegment a = ar.allocate(aOff + (long) n * k * 4);
            MemorySegment c = ar.allocate(cOff + (long) m * n * 4);
            MemorySegment.copy(W, 0, w.asSlice(wOff), JAVA_FLOAT, 0, m * k);
            MemorySegment.copy(A, 0, a.asSlice(aOff), JAVA_FLOAT, 0, n * k);
            assertEquals(JAM.OK, JAM.mm(w, wOff, JAM.F32, k, a, aOff, JAM.F32, k, c, cOff, JAM.F32, m, m, n, k));
            float[] out = new float[m * n];
            MemorySegment.copy(c.asSlice(cOff), JAVA_FLOAT, 0, out, 0, m * n);
            assertArrayEquals(refMM(W, A, m, n, k), out, 1e-3f);
        }
    }

    @Test
    void jniAndFfmBackendsAgree() {                     // both bindings reach the same native jam_mm, bit-identical
        try (Arena ar = Arena.ofConfined()) {
            int m = 5, n = 3, k = 32;
            float[] W = new float[m * k], A = new float[n * k];
            for (int i = 0; i < W.length; i++) W[i] = (float) Math.sin(i * 0.17);
            for (int i = 0; i < A.length; i++) A[i] = (float) Math.cos(i * 0.11);
            MemorySegment w = ar.allocate((long) m * k * 4), a = ar.allocate((long) n * k * 4);
            MemorySegment.copy(W, 0, w, JAVA_FLOAT, 0, m * k);
            MemorySegment.copy(A, 0, a, JAVA_FLOAT, 0, n * k);
            MemorySegment cj = ar.allocate((long) m * n * 4), cf = ar.allocate((long) m * n * 4);
            int stj = JAM.mmJni(w.address(), JAM.F32, k, a.address(), JAM.F32, k, cj.address(), JAM.F32, m, m, n, k);
            int stf = JAM.mmFfm(w.address(), JAM.F32, k, a.address(), JAM.F32, k, cf.address(), JAM.F32, m, m, n, k);
            assertEquals(JAM.OK, stj, "JNI backend");
            assertEquals(JAM.OK, stf, "FFM (Panama) backend");
            for (int i = 0; i < m * n; i++)
                assertEquals(cj.getAtIndex(JAVA_FLOAT, i), cf.getAtIndex(JAVA_FLOAT, i), 0f,
                        "JNI and FFM must be identical");
        }
    }

    @Test
    void concurrentCallsHitEbusy() throws Exception {   // global ctx is a serial stream: collisions -> EBUSY, never corruption
        int m = 256, n = 256, k = 256;                  // ~0.3ms/call — window wide enough to overlap concurrent callers
        int threads = 4, rounds = 20;
        AtomicInteger ok = new AtomicInteger(), ebusy = new AtomicInteger(), other = new AtomicInteger();
        try (Arena ar = Arena.ofShared()) {             // shared: segments touched from worker threads
            MemorySegment w = ar.allocate((long) m * k * 4), a = ar.allocate((long) n * k * 4);
            for (int i = 0; i < m * k; i++) w.setAtIndex(JAVA_FLOAT, i, (float) Math.sin(i * 0.001));
            for (int i = 0; i < n * k; i++) a.setAtIndex(JAVA_FLOAT, i, (float) Math.cos(i * 0.001));
            MemorySegment[] cs = new MemorySegment[threads];
            for (int t = 0; t < threads; t++) cs[t] = ar.allocate((long) m * n * 4);   // per-thread output, no aliasing
            ExecutorService pool = Executors.newFixedThreadPool(threads);
            CyclicBarrier barrier = new CyclicBarrier(threads);
            List<Future<?>> futs = new ArrayList<>();
            for (int t = 0; t < threads; t++) {
                final MemorySegment c = cs[t];
                futs.add(pool.submit(() -> {
                    for (int r = 0; r < rounds; r++) {
                        try { barrier.await(); } catch (Exception e) { throw new RuntimeException(e); }  // fire ~simultaneously
                        int st = JAM.mm(w, 0, JAM.F32, k, a, 0, JAM.F32, k, c, 0, JAM.F32, m, m, n, k);
                        (st == JAM.OK ? ok : st == JAM.EBUSY ? ebusy : other).incrementAndGet();
                    }
                }));
            }
            for (Future<?> f : futs) f.get();
            pool.shutdown();
        }
        assertEquals(0, other.get(), "every concurrent call returns OK or EBUSY, nothing else");
        assertTrue(ok.get() > 0, "some calls win the serial stream");
        assertTrue(ebusy.get() > 0, "concurrent calls on the global context hit the EBUSY guard");
    }

    @Test
    void configResolvesPropertyThenDefault() {          // -Dprop, else env (JAM_BINDING form), else default
        System.setProperty("jam.unit.test.knob", "fromProp");
        try {
            assertEquals("fromProp", JAM.config("jam.unit.test.knob", "def"), "system property wins");
            assertEquals("def", JAM.config("jam.unit.test.absent", "def"), "absent -> default (env form jam.x.y -> JAM_X_Y)");
        } finally {
            System.clearProperty("jam.unit.test.knob");
        }
    }

    @Test
    void ggmlCodesMatchPublicTags() throws Exception {   // structural invariant — auto-covers new dtypes
        for (GGMLType f : GGMLType.values()) {
            assertEquals(f, GGMLType.byCode(f.ggml), () -> f + ": byCode round-trip");                       // code -> enum
            assertEquals(f.ggml, JAM.class.getField(f.name()).getInt(null), () -> f + ": public int tag");  // enum <-> JAM.<name>
        }
    }
}
