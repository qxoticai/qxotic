package com.qxotic.jinfer;

import com.qxotic.format.gguf.GGMLType;
import com.qxotic.jam.JAM;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Routes each matmul to the fastest applicable backend, with {@link ScalarMatMul} as the universal
 * floor - so {@code mm} is total ({@code void}). The policy is the measured one:
 *
 * <ul>
 *   <li><b>prefill</b> (n&gt;1, compute-bound): jam → Vector tile → scalar.
 *   <li><b>decode</b> (n==1, bandwidth-bound): Vector matvec (when the JIT compiles it well; C2
 *       runs the k-quant dots un-intrinsified, so those go to jam there) → jam → scalar.
 * </ul>
 *
 * All the capability gates (F32 operands, dtype, alignment, vector width) live here, so the
 * backends are only ever called when applicable and never have to decline. (jam can still decline
 * at runtime on EBUSY; it absorbs that into its own scalar fallback.)
 */
final class Dispatch implements MatMul {

    private static final System.Logger LOG = System.getLogger("jinfer.jam");

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
        List<JAM.Provider> providers = JAM.providers();
        // Both fast backends are the same JamMatMul adapter over a different JAM; each declines to
        // the floor.
        JAM vectorJam = loadJam(providers, "vector");
        MatMul vector = vectorJam != null ? new JamMatMul(vectorJam, scalar) : null;
        JAM nativeJam = boolFlag("jinfer.disableJam") ? null : loadJam(providers, "native");
        // A native runtime decline (EBUSY, or an older libjam without a dtype's kernel) falls to
        // the vector tile when one exists. The vector backend re-gates per call and itself declines
        // to the floor, so a stale native library degrades to the fast Java path, never the scalar
        // floor.
        MatMul jam =
                nativeJam != null
                        ? new JamMatMul(nativeJam, vector != null ? vector : scalar)
                        : null;
        return new Dispatch(jam, vector, scalar);
    }

    /**
     * The named JAM backend, or {@code null} if absent/unavailable. Selection by id is jinfer
     * policy; jam-core only discovers available providers.
     */
    private static JAM loadJam(List<JAM.Provider> providers, String id) {
        for (JAM.Provider provider : providers) {
            if (!provider.id().equals(id)) continue;
            try {
                return provider.create();
            } catch (Throwable t) {
                LOG.log(System.Logger.Level.WARNING, "jam {0} backend unavailable ({1})", id, t);
                return null;
            }
        }
        return null;
    }

    /**
     * Strict boolean system property: true/false (any case) parse; anything else warns and reads as
     * false - Boolean.getBoolean silently treats "1"/"yes"/typos as false, which has already cost a
     * mis-measured benchmark.
     */
    private static boolean boolFlag(String name) {
        String v = System.getProperty(name);
        if (v == null) return false;
        if (v.equalsIgnoreCase("true")) return true;
        if (v.equalsIgnoreCase("false")) return false;
        LOG.log(
                System.Logger.Level.WARNING,
                "ignoring -D{0}={1} (expected true or false)",
                name,
                v);
        return false;
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
            // decode matvec: the scalar floor's dot() vectorizes per row in parallel - measured
            // identical to the old specialized Vector gemv on this memory-bound kernel. Exception:
            // C2 runs the byte-unpack-heavy k-quant dots through the Vector API's un-intrinsified
            // fallback (Q4_K_M decode 11 t/s vs jam's 31), so those go to jam there; the dense
            // dots stay Java even on C2 (Q8_0 measured 42 via dot vs 29 via jam).
            boolean slowDot = !FloatTensor.FAST_VECTOR_JIT && bytePackedDot(t);
            chosen =
                    FloatTensor.USE_VECTOR_API && !slowDot
                            ? scalar
                            : (jam != null && f32io && jamSupports(t, k) ? jam : scalar);
        } else {
            // Measured prefill policy: the native repack/VNNI band kernels beat the Java tile for
            // every dtype, Q1_0 included since its packed-sign-bit 16-row VNNI band landed
            // (jam_q1_0_repack_band; the earlier per-row vec_dot lost ~15% to the Java tile and
            // was demoted to the sub-band rungs).
            chosen =
                    jam != null && f32io && jamSupports(t, k)
                            ? jam
                            : vector != null && f32io && gemmApplies(t, k, wOff) ? vector : scalar;
        }
        chosen.mm(w, wOff, wStride, a, aOff, aStride, c, cOff, cStride, m, n, k);
    }

    /**
     * dtypes whose vector dot C2 executes largely un-intrinsified (the k-quants' byte shift/or/sub
     * unpack chains; measured Q4_K_M decode collapse). Measured non-members: Q4_0's single-nibble
     * unpack is fine on C2 (llama-1B tg 114 vs Graal's 118), and MXFP4 loses ~20% on C2 but jam
     * routing loses more (gpt-oss-20b tg 22.8 Java dot vs 18.0 jam - MoE decode issues ~24k tiny
     * expert gemvs per pass and each jam call pays the FFM boundary). NVFP4 is structurally exempt:
     * its dot dequantizes with scalar code then runs a dense F32 vector dot, so there is no
     * byte-vector unpack for C2 to fall back on (kernel probe: 1.8x hot-cache gap, from the scalar
     * decode loop's codegen).
     */
    private static boolean bytePackedDot(GGMLType t) {
        return switch (t) {
            case Q4_K, Q5_K, Q6_K -> true;
            default -> false;
        };
    }

    /**
     * dtypes jam has a kernel for (it enforces exact alignment and absorbs a mismatch via its
     * fallback); the alignment is the dtype's own block size (1 for the dense float types).
     */
    private static boolean jamSupports(GGMLType t, int k) {
        return switch (t) {
            case Q8_0, Q4_0, Q4_K, Q5_K, Q6_K, MXFP4, NVFP4, Q1_0, F16, BF16, F32 ->
                    k % t.getElementsPerBlock() == 0;
            default -> false;
        };
    }

    /**
     * "vectors present AND 512-bit" - the precondition for the Vector prefill tile (constant,
     * JIT-folded).
     */
    private static final boolean IS_512 =
            FloatTensor.USE_VECTOR_API && FloatTensor.F_SPECIES.vectorBitSize() == 512;

    /** dtypes with a register-tiled Vector prefill kernel (the rest fall to the scalar floor). */
    private static boolean hasGemmTile(GGMLType t) {
        return switch (t) {
            case Q8_0, Q4_0, Q4_K, Q5_K, Q6_K, MXFP4, NVFP4, Q1_0 -> true;
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
