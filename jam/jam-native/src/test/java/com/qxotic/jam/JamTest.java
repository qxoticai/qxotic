package com.qxotic.jam;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.function.Executable;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
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
 * Tests the jam Java API end to end: the native library loads (via NativeLoader), and {@code NativeJAM.global().mm}
 * computes R = W @ Aᵀ correctly for F32/F16/Q8_0 weights, bounds-checks its native MemorySegments, and reports
 * the right status for bad/unsupported calls. Operands are off-heap. The suite runs once per binding
 * (jni + ffm) via two surefire passes; the ffm one sets -Djam.native.binding=ffm (see pom.xml).
 */
class JamTest {

    private static final JAM jam = NativeJAM.global();

    /** F32 matmul through the API; returns C [m×n] token-major (out[j*m + i]). */
    private static float[] mmF32(Arena ar, float[] W, float[] A, int m, int n, int k) {
        MemorySegment w = ar.allocate((long) m * k * Float.BYTES);
        MemorySegment a = ar.allocate((long) n * k * Float.BYTES);
        MemorySegment c = ar.allocate((long) m * n * Float.BYTES);
        MemorySegment.copy(W, 0, w, JAVA_FLOAT, 0, m * k);
        MemorySegment.copy(A, 0, a, JAVA_FLOAT, 0, n * k);
        assertEquals(JAM.OK, jam.mm(w, a, c, JAM.F32, m, n, k), "jam.mm should return OK");
        float[] out = new float[m * n];
        MemorySegment.copy(c, JAVA_FLOAT, 0, out, 0, m * n);
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

    /** assert the bounds-checked mm() rejects this call in JAVA (IndexOutOfBoundsException), never reaching native. */
    private static void rejects(String why, Executable mmCall) {
        assertThrows(IndexOutOfBoundsException.class, mmCall, why);
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
    void fullStridedMatchesContiguous() {            // the 7-arg shortcut == the 15-arg full form, contiguous
        try (Arena ar = Arena.ofConfined()) {
            int m = 5, n = 3, k = 16;
            float[] W = new float[m * k], A = new float[n * k];
            for (int i = 0; i < W.length; i++) W[i] = (float) Math.sin(i * 0.3);
            for (int i = 0; i < A.length; i++) A[i] = (float) Math.cos(i * 0.2);
            MemorySegment w = ar.allocate((long) m * k * 4), a = ar.allocate((long) n * k * 4), c = ar.allocate((long) m * n * 4);
            MemorySegment.copy(W, 0, w, JAVA_FLOAT, 0, m * k);
            MemorySegment.copy(A, 0, a, JAVA_FLOAT, 0, n * k);
            assertEquals(JAM.OK, jam.mm(w, a, c, JAM.F32, m, n, k));   // shortcut: offsets 0, strides k/k/m, F32
            float[] out = new float[m * n];
            MemorySegment.copy(c, JAVA_FLOAT, 0, out, 0, m * n);
            assertArrayEquals(refMM(W, A, m, n, k), out, 1e-3f);
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
            assertEquals(JAM.OK, jam.mm(w, a, c, JAM.Q8_0, m, n, k));

            float[] ref = refMM(wdq, A, m, n, k);    // token-major; kernel may requant A -> loose tol
            for (int i = 0; i < m * n; i++)
                assertEquals(ref[i], c.getAtIndex(JAVA_FLOAT, i), 0.05f + 0.05f * Math.abs(ref[i]),
                        "Q8_0 result within int8 activation-requant tolerance");
        }
    }

    @Test
    void denseHalfWeightsSupported() {               // F16/BF16 dense weights run (AVX-512 tile, else float floor)
        try (Arena ar = Arena.ofConfined()) {
            int m = 4, n = 2, k = 16;
            MemorySegment w = ar.allocate((long) m * k * 2);                 // 2 bytes/elem
            MemorySegment a = ar.allocate((long) n * k * 4), c = ar.allocate((long) m * n * 4);
            assertEquals(JAM.OK, jam.mm(w, a, c, JAM.F16,  m, n, k), "F16 weights");
            assertEquals(JAM.OK, jam.mm(w, a, c, JAM.BF16, m, n, k), "BF16 weights");
        }
    }

    @Test
    void rejectsHeapSegments() {                     // array-backed segments have no native address -> reject up front
        try (Arena ar = Arena.ofConfined()) {
            int m = 2, n = 2, k = 32;
            MemorySegment a = ar.allocate((long) n * k * 4), c = ar.allocate((long) m * n * 4);
            MemorySegment heapW = MemorySegment.ofArray(new float[m * k]);   // on-heap
            assertThrows(IllegalArgumentException.class,
                    () -> jam.mm(heapW, a, c, JAM.F32, m, n, k), "heap weight");
            MemorySegment w = ar.allocate((long) m * k * 4);
            assertThrows(IllegalArgumentException.class,
                    () -> jam.mm(w, MemorySegment.ofArray(new float[n * k]), c, JAM.F32, m, n, k), "heap activation");
        }
    }

    @Test
    void boundsRejectUndersizedOperands() {          // each operand one byte short -> throws; exact-fit -> OK
        try (Arena ar = Arena.ofConfined()) {
            int m = 4, n = 3, k = 64;
            long wB = (long) m * k * 4, aB = (long) n * k * 4, cB = (long) m * n * 4;
            MemorySegment w = ar.allocate(wB), a = ar.allocate(aB), c = ar.allocate(cB);
            assertEquals(JAM.OK, jam.mm(w, 0, JAM.F32, k, a, 0, JAM.F32, k, c, 0, JAM.F32, m, m, n, k), "exact fit, no false positive");
            rejects("weight -1B",     () -> jam.mm(ar.allocate(wB - 1), 0, JAM.F32, k, a, 0, JAM.F32, k, c, 0, JAM.F32, m, m, n, k));
            rejects("activation -1B", () -> jam.mm(w, 0, JAM.F32, k, ar.allocate(aB - 1), 0, JAM.F32, k, c, 0, JAM.F32, m, m, n, k));
            rejects("result -1B",     () -> jam.mm(w, 0, JAM.F32, k, a, 0, JAM.F32, k, ar.allocate(cB - 1), 0, JAM.F32, m, m, n, k));
        }
    }

    @Test
    void boundsRejectBadOffsets() {                  // a byte offset must be counted + must be non-negative
        try (Arena ar = Arena.ofConfined()) {
            int m = 4, n = 3, k = 64;
            MemorySegment w = ar.allocate((long) m * k * 4), a = ar.allocate((long) n * k * 4), c = ar.allocate((long) m * n * 4);
            rejects("weight offset overflow", () -> jam.mm(w, 4, JAM.F32, k, a, 0, JAM.F32, k, c, 0, JAM.F32, m, m, n, k));
            rejects("act offset overflow",    () -> jam.mm(w, 0, JAM.F32, k, a, 4, JAM.F32, k, c, 0, JAM.F32, m, m, n, k));
            rejects("res offset overflow",    () -> jam.mm(w, 0, JAM.F32, k, a, 0, JAM.F32, k, c, 4, JAM.F32, m, m, n, k));
            rejects("negative offset",        () -> jam.mm(w, -4, JAM.F32, k, a, 0, JAM.F32, k, c, 0, JAM.F32, m, m, n, k));
        }
    }

    @Test
    void boundsHonorStrides() {                      // ldw/lda/ldr are ELEMENT strides; size = (rows-1)*stride + lastRow
        try (Arena ar = Arena.ofConfined()) {
            int m = 4, n = 3, k = 64, pad = 8;
            MemorySegment w = ar.allocate((long) m * k * 4), a = ar.allocate((long) n * k * 4), c = ar.allocate((long) m * n * 4);
            // strided weight (ldw = k+pad)
            int ldw = k + pad; long wN = (long) (m - 1) * ldw * 4 + (long) k * 4;
            assertEquals(JAM.OK, jam.mm(ar.allocate(wN), 0, JAM.F32, ldw, a, 0, JAM.F32, k, c, 0, JAM.F32, m, m, n, k), "strided weight exact");
            rejects("strided weight -1B", () -> jam.mm(ar.allocate(wN - 1), 0, JAM.F32, ldw, a, 0, JAM.F32, k, c, 0, JAM.F32, m, m, n, k));
            // strided activation (lda = k+pad)
            int lda = k + pad; long aN = (long) (n - 1) * lda * 4 + (long) k * 4;
            assertEquals(JAM.OK, jam.mm(w, 0, JAM.F32, k, ar.allocate(aN), 0, JAM.F32, lda, c, 0, JAM.F32, m, m, n, k), "strided act exact");
            rejects("strided act -1B", () -> jam.mm(w, 0, JAM.F32, k, ar.allocate(aN - 1), 0, JAM.F32, lda, c, 0, JAM.F32, m, m, n, k));
            // padded result (ldr = m+pad) — the token-major R stride (jinfer's max-dim buffers)
            int ldr = m + pad; long cN = (long) (n - 1) * ldr * 4 + (long) m * 4;
            assertEquals(JAM.OK, jam.mm(w, 0, JAM.F32, k, a, 0, JAM.F32, k, ar.allocate(cN), 0, JAM.F32, ldr, m, n, k), "padded result exact");
            rejects("padded result -1B", () -> jam.mm(w, 0, JAM.F32, k, a, 0, JAM.F32, k, ar.allocate(cN - 1), 0, JAM.F32, ldr, m, n, k));
        }
    }

    @Test
    void boundsBlockAwareAndDense() {                // weight bytes are block-aware (quant) / 2-per-elem (F16/BF16)
        try (Arena ar = Arena.ofConfined()) {
            int m = 4, n = 3, k = 256;               // k%256 so Q4_K is legal
            MemorySegment a = ar.allocate((long) n * k * 4), c = ar.allocate((long) m * n * 4);
            long q8 = (long) m * (k / 32) * 34, q4k = (long) m * (k / 256) * 144, half = (long) m * k * 2;
            assertEquals(JAM.OK, jam.mm(ar.allocate(q8), 0, JAM.Q8_0, k, a, 0, JAM.F32, k, c, 0, JAM.F32, m, m, n, k), "Q8_0 exact");
            rejects("Q8_0 -1B", () -> jam.mm(ar.allocate(q8 - 1), 0, JAM.Q8_0, k, a, 0, JAM.F32, k, c, 0, JAM.F32, m, m, n, k));
            rejects("Q4_K -1B", () -> jam.mm(ar.allocate(q4k - 1), 0, JAM.Q4_K, k, a, 0, JAM.F32, k, c, 0, JAM.F32, m, m, n, k));
            rejects("F16 -1B",  () -> jam.mm(ar.allocate(half - 1), 0, JAM.F16, k, a, 0, JAM.F32, k, c, 0, JAM.F32, m, m, n, k));
            rejects("BF16 -1B", () -> jam.mm(ar.allocate(half - 1), 0, JAM.BF16, k, a, 0, JAM.F32, k, c, 0, JAM.F32, m, m, n, k));
        }
    }

    @Test
    void nativeClassifiesUnsupportedAndDegenerate() {   // safe mm must NOT throw where native returns a clean status
        try (Arena ar = Arena.ofConfined()) {
            MemorySegment w = ar.allocate(8192), a = ar.allocate(8192), c = ar.allocate(8192);  // generous; bounds never bite
            int m = 2, n = 2, k = 64;
            int Q2_K = 10;   // a recognized-but-unsupported GGUF tag (not a JAM.* constant)
            assertEquals(JAM.EUNSUPPORTED, jam.mm(w, 0, Q2_K,    k, a, 0, JAM.F32, k, c, 0, JAM.F32, m, m, n, k), "unsupported weight dtype");
            assertEquals(JAM.EUNSUPPORTED, jam.mm(w, 0, JAM.F32, k, a, 0, JAM.F16, k, c, 0, JAM.F32, m, m, n, k), "non-F32 activation");
            assertEquals(JAM.EUNSUPPORTED, jam.mm(w, 0, JAM.F32, k, a, 0, JAM.F32, k, c, 0, JAM.F16, m, m, n, k), "non-F32 result");
            assertEquals(JAM.EUNSUPPORTED, jam.mm(w, 0, JAM.Q8_0, 48, a, 0, JAM.F32, 48, c, 0, JAM.F32, m, m, n, 48), "k not a Q8_0-block multiple");
            assertEquals(JAM.EINVAL,       jam.mm(w, 0, JAM.F32, k, a, 0, JAM.F32, k, c, 0, JAM.F32, m, m, 0, k), "n <= 0 (guard skips bounds)");
            assertEquals(JAM.EINVAL,       jam.mm(w, 0, JAM.F32, k - 1, a, 0, JAM.F32, k, c, 0, JAM.F32, m, m, n, k), "ldw < k (guard skips bounds)");
        }
    }

    @Test
    void nonZeroOffsetMatmul() {                     // sub-segment placement: address()+off arithmetic + bounds
        try (Arena ar = Arena.ofConfined()) {
            int m = 3, n = 2, k = 32;
            long wOff = 64, aOff = 32, cOff = 16;    // byte offsets, all 4-aligned
            float[] W = new float[m * k], A = new float[n * k];
            for (int i = 0; i < W.length; i++) W[i] = (float) Math.sin(i * 0.21);
            for (int i = 0; i < A.length; i++) A[i] = (float) Math.cos(i * 0.13);
            MemorySegment w = ar.allocate(wOff + (long) m * k * 4);
            MemorySegment a = ar.allocate(aOff + (long) n * k * 4);
            MemorySegment c = ar.allocate(cOff + (long) m * n * 4);
            MemorySegment.copy(W, 0, w.asSlice(wOff), JAVA_FLOAT, 0, m * k);
            MemorySegment.copy(A, 0, a.asSlice(aOff), JAVA_FLOAT, 0, n * k);
            assertEquals(JAM.OK, jam.mm(w, wOff, JAM.F32, k, a, aOff, JAM.F32, k, c, cOff, JAM.F32, m, m, n, k));
            float[] out = new float[m * n];
            MemorySegment.copy(c.asSlice(cOff), JAVA_FLOAT, 0, out, 0, m * n);
            assertArrayEquals(refMM(W, A, m, n, k), out, 1e-3f);
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
                    for (int rd = 0; rd < rounds; rd++) {
                        try { barrier.await(); } catch (Exception e) { throw new RuntimeException(e); }  // fire ~simultaneously
                        int st = jam.mm(w, a, c, JAM.F32, m, n, k);
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
    void configResolvesPropertyThenDefault() {       // -Dprop, else env (jam.x.y -> JAM_X_Y), else default
        System.setProperty("jam.unit.test.knob", "fromProp");
        try {
            assertEquals("fromProp", NativeLoader.config("jam.unit.test.knob", "def"), "system property wins");
            assertEquals("def", NativeLoader.config("jam.unit.test.absent", "def"), "absent -> default (env form jam.x.y -> JAM_X_Y)");
        } finally {
            System.clearProperty("jam.unit.test.knob");
        }
    }

    @Test
    void ggmlCodesMatchPublicTags() throws Exception {   // structural invariant: internal enum <-> public int tags
        for (GGMLType g : GGMLType.values()) {
            assertEquals(g, GGMLType.byCode(g.ggml), () -> g + ": byCode round-trip");                       // code -> enum
            assertEquals(g.ggml, JAM.class.getField(g.name()).getInt(null), () -> g + ": public int tag");   // enum <-> JAM.<name>
        }
    }
}
