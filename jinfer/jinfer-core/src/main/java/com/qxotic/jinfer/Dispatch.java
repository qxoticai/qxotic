package com.qxotic.jinfer;

import com.qxotic.format.gguf.GGMLType;

/**
 * Routes each matmul to the fastest applicable backend, with {@link ScalarMatMul} as the universal floor —
 * so {@code mm} is total ({@code void}). The policy is the measured one:
 * <ul>
 *   <li><b>prefill</b> (n&gt;1, compute-bound): jam → Vector tile → scalar.
 *   <li><b>decode</b> (n==1, bandwidth-bound): Vector matvec (when vectors) → jam (when none) → scalar.
 * </ul>
 * All the capability gates (F32 operands, dtype, alignment, vector width) live here, so the backends are
 * only ever called when applicable and never have to decline. (jam can still decline at runtime on EBUSY;
 * it absorbs that into its own scalar fallback.)
 */
final class Dispatch implements MatMul {

    private final MatMul jam;       // null if jam couldn't load
    private final MatMul vector;
    private final MatMul scalar;

    private Dispatch(MatMul jam, MatMul vector, MatMul scalar) {
        this.jam = jam;
        this.vector = vector;
        this.scalar = scalar;
    }

    static Dispatch create() {
        MatMul scalar = new ScalarMatMul();
        MatMul vector = new VectorMatMul();
        MatMul jam = JamMatMul.tryLoad() ? new JamMatMul(scalar) : null;   // jam falls back to the floor
        return new Dispatch(jam, vector, scalar);
    }

    // --- per-shape matmul byte attribution (-Djinfer.mmTrace) ---
    static final boolean MM_TRACE = System.getProperty("jinfer.mmTrace") != null;
    private static final java.util.Map<String, long[]> MM_HIST = new java.util.concurrent.ConcurrentHashMap<>();
    static {
        if (MM_TRACE) Runtime.getRuntime().addShutdownHook(new Thread(Dispatch::mmDump));
    }
    private static void mmRecord(GGMLType t, int m, int n, int k) {
        long bytes = (long) m * k * t.getBlockByteSize() / t.getElementsPerBlock();
        String key = String.format("%-6s %6dx%-6d n=%d", t, m, k, n);
        long[] e = MM_HIST.computeIfAbsent(key, x -> new long[2]);
        synchronized (e) { e[0]++; e[1] += bytes; }
    }
    static void mmDump() {
        long tot = 0, totN1 = 0;
        System.err.println("=== matmul byte attribution (count, total weight bytes) ===");
        var sorted = new java.util.ArrayList<>(MM_HIST.entrySet());
        sorted.sort((a, b) -> Long.compare(b.getValue()[1], a.getValue()[1]));
        for (var e : sorted) {
            long[] v = e.getValue();
            System.err.printf("  %-28s  x%-5d  %8.1f MB%n", e.getKey(), v[0], v[1] / 1e6);
            tot += v[1];
            if (e.getKey().endsWith("n=1")) totN1 += v[1];
        }
        System.err.printf("  TOTAL %.2f GB  (n=1 decode: %.2f GB)%n", tot / 1e9, totN1 / 1e9);
    }

    @Override
    public void mm(FloatTensor w, long wOff, int wStride,
                   FloatTensor a, long aOff, int aStride,
                   FloatTensor c, long cOff, int cStride,
                   int m, int n, int k) {
        GGMLType t = w.type();
        if (MM_TRACE) mmRecord(t, m, n, k);
        boolean f32io = a instanceof F32FloatTensor && c instanceof F32FloatTensor && a != c;

        // Pick the backend, then issue one matmul. Decode (n==1): Vector matvec when vectors are present
        // (the scalar floor's dot() still vectorizes for what Vector skips), jam when there are none.
        // Prefill: jam, else the Java Vector tile, else the floor.
        MatMul chosen;
        if (n == 1) {
            chosen = FloatTensor.USE_VECTOR_API
                    ? (f32io && VectorMatMul.gemvApplies(t, m, k, wOff) ? vector : scalar)
                    : (jam != null && f32io && jamSupports(t, k) ? jam : scalar);
        } else {
            chosen = jam != null && f32io && jamSupports(t, k) ? jam
                   : f32io && VectorMatMul.gemmApplies(t, k, wOff) ? vector
                   : scalar;
        }
        chosen.mm(w, wOff, wStride, a, aOff, aStride, c, cOff, cStride, m, n, k);
    }

    /** dtypes jam has a kernel for (it enforces exact alignment and absorbs a mismatch via its fallback);
     *  the alignment is the dtype's own block size (1 for the dense float types). */
    private static boolean jamSupports(GGMLType t, int k) {
        return switch (t) {
            case Q8_0, Q4_0, Q4_K, Q5_K, Q6_K, MXFP4, F16, BF16, F32 -> k % t.getElementsPerBlock() == 0;
            default -> false;
        };
    }
}
