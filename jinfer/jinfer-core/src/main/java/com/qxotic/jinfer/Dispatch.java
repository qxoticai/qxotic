package com.qxotic.jinfer;

import com.qxotic.format.gguf.GGMLType;
import com.qxotic.jam.JAM;
import com.qxotic.jam.NativeJAM;
import com.qxotic.jam.VectorJAM;
import java.util.ArrayList;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Routes each matmul to the fastest applicable backend, with {@link ScalarMatMul} as the universal
 * floor — so {@code mm} is total ({@code void}). The policy is the measured one:
 *
 * <ul>
 *   <li><b>prefill</b> (n&gt;1, compute-bound): jam → Vector tile → scalar.
 *   <li><b>decode</b> (n==1, bandwidth-bound): Vector matvec (when vectors) → jam (when none) →
 *       scalar.
 * </ul>
 *
 * All the capability gates (F32 operands, dtype, alignment, vector width) live here, so the
 * backends are only ever called when applicable and never have to decline. (jam can still decline
 * at runtime on EBUSY; it absorbs that into its own scalar fallback.)
 */
final class Dispatch implements MatMul {

    private final MatMul jam; // native jam,     or null if libjam couldn't load
    private final MatMul vector; // Vector API jam,  or null if jdk.incubator.vector is absent
    private final MatMul scalar; // universal floor (jinfer-native dot)

    private Dispatch(MatMul jam, MatMul vector, MatMul scalar) {
        this.jam = jam;
        this.vector = vector;
        this.scalar = scalar;
    }

    static final MatMul ACTIVE = create();

    static Dispatch create() {
        MatMul scalar = new ScalarMatMul();
        // Both fast backends are the same JamMatMul adapter over a different JAM; each declines to
        // the floor.
        MatMul vector = VectorJAM.isAvailable() ? new JamMatMul(new VectorJAM(), scalar) : null;
        JAM nativeJam = loadNative();
        MatMul jam = nativeJam != null ? new JamMatMul(nativeJam, scalar) : null;
        return new Dispatch(jam, vector, scalar);
    }

    /**
     * The native jam backend, or {@code null} if unavailable / disabled. Direct touch triggers
     * NativeJAM's static init (libjam load) — deliberately NOT Class.forName, whose reflective
     * lookup needs registration on native image and would silently disable jam there.
     */
    private static JAM loadNative() {
        if (Boolean.getBoolean("jinfer.disableJam"))
            return null; // force the Java backends (testing)
        try {
            return NativeJAM.global();
        } catch (Throwable t) {
            System.err.println(
                    "jam native library unavailable (" + t + "); using the Java backends.");
            return null;
        }
    }

    // --- per-shape matmul byte attribution (-Djinfer.mmTrace) ---
    static final boolean MM_TRACE = System.getProperty("jinfer.mmTrace") != null;
    private static final Map<String, long[]> MM_HIST = new ConcurrentHashMap<>();

    static {
        if (MM_TRACE) Runtime.getRuntime().addShutdownHook(new Thread(Dispatch::mmDump));
    }

    private static void mmRecord(GGMLType t, int m, int n, int k) {
        long bytes = (long) m * k * t.getBlockByteSize() / t.getElementsPerBlock();
        String key = String.format("%-6s %6dx%-6d n=%d", t, m, k, n);
        long[] e = MM_HIST.computeIfAbsent(key, x -> new long[2]);
        synchronized (e) {
            e[0]++;
            e[1] += bytes;
        }
    }

    static void mmDump() {
        long tot = 0, totN1 = 0;
        System.err.println("=== matmul byte attribution (count, total weight bytes) ===");
        var sorted = new ArrayList<>(MM_HIST.entrySet());
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
    public void mm(
            FloatTensor w,
            long wOff,
            int wStride,
            FloatTensor a,
            long aOff,
            int aStride,
            FloatTensor c,
            long cOff,
            int cStride,
            int m,
            int n,
            int k) {
        GGMLType t = w.type();
        if (MM_TRACE) mmRecord(t, m, n, k);
        boolean f32io = a instanceof F32FloatTensor && c instanceof F32FloatTensor && a != c;

        // Pick the backend, then issue one matmul. Decode (n==1): the scalar floor's parallel
        // one-row dot()
        // (it vectorizes) when there's a Vector API, jam when there isn't. Prefill: jam, else
        // Vector tile, else floor.
        MatMul chosen;
        if (n == 1) {
            // decode matvec: the scalar floor's dot() vectorizes per row in parallel — measured
            // identical to
            // the old specialized Vector gemv on this memory-bound kernel. jam only when there's no
            // Vector API.
            chosen =
                    FloatTensor.USE_VECTOR_API
                            ? scalar
                            : (jam != null && f32io && jamSupports(t, k) ? jam : scalar);
        } else {
            chosen =
                    jam != null && f32io && jamSupports(t, k)
                            ? jam
                            : vector != null && f32io && gemmApplies(t, k, wOff) ? vector : scalar;
        }
        chosen.mm(w, wOff, wStride, a, aOff, aStride, c, cOff, cStride, m, n, k);
    }

    /**
     * dtypes jam has a kernel for (it enforces exact alignment and absorbs a mismatch via its
     * fallback); the alignment is the dtype's own block size (1 for the dense float types).
     */
    private static boolean jamSupports(GGMLType t, int k) {
        return switch (t) {
            case Q8_0, Q4_0, Q4_K, Q5_K, Q6_K, MXFP4, NVFP4, F16, BF16, F32 ->
                    k % t.getElementsPerBlock() == 0;
            default -> false;
        };
    }

    /**
     * "vectors present AND 512-bit" — the precondition for the Vector prefill tile (constant,
     * JIT-folded).
     */
    private static final boolean IS_512 =
            FloatTensor.USE_VECTOR_API && FloatTensor.F_SPECIES.vectorBitSize() == 512;

    /** dtypes with a register-tiled Vector prefill kernel (the rest fall to the scalar floor). */
    private static boolean hasGemmTile(GGMLType t) {
        return switch (t) {
            case Q8_0, Q4_0, Q4_K, Q5_K, Q6_K, MXFP4, NVFP4 -> true;
            default -> false; // F16, BF16, F32 -> dot floor
        };
    }

    /**
     * Whether the Vector prefill tile applies: 512-bit vectors, a tileable dtype, block-aligned k +
     * weight offset.
     */
    private static boolean gemmApplies(GGMLType t, int k, long wOff) {
        if (!IS_512 || !hasGemmTile(t)) return false;
        int blk = t.getElementsPerBlock();
        return (k % blk == 0) && (wOff % blk == 0);
    }
}
