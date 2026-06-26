package com.qxotic.jam;

import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.lang.ref.Reference;
import java.util.Locale;

import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

/**
 * jam — fastest multithreaded CPU matmul: {@code C = A @ Bᵀ}, with quantized weights.
 *
 * <p>The Java surface runs on the <b>process-global</b> jam context, created lazily on first use and
 * configured through environment variables ({@code JAM_NUM_THREADS}, {@code JAM_ISA}); no handle to manage.
 *
 * <p><b>One public entry point</b> — {@link #mm(MemorySegment, long, int, int, MemorySegment, long, int, int, MemorySegment, long, int, int, int, int, int)}:
 * each operand is a {@link MemorySegment} + byte offset, kept reachable across the call and BOUNDS-CHECKED
 * against what the kernel touches, so a bad size throws in Java rather than corrupting native memory. There is
 * deliberately no public raw-address variant — the surface is bullet-proof by construction.
 *
 * <p>Internally, two package-private backends reach the same native {@code jam_mm}, selected once via
 * {@code -Djam.binding} (or {@code JAM_BINDING}): {@code jni} (default — the {@code jam_jni.c} shim) or
 * {@code ffm} (a Panama downcall). The selector is a {@code static final} boolean the JIT folds away.
 *
 * <p><b>Concurrency:</b> the global context is a single serial execution stream — multithreaded <i>within</i>
 * a call, but calls must come from <b>one</b> thread (concurrent calls return {@code EBUSY}).
 */
public final class JAM {

    // ---- type tags: numerically identical to GGML's ggml_type (append-only ABI; never reorder).
    //      The package-private GGMLType enum mirrors these with block geometry (a test keeps the two in sync). ----
    public static final int F32 = 0, F16 = 1,
            Q4_0 = 2, Q4_1 = 3, Q5_0 = 6, Q5_1 = 7, Q8_0 = 8,
            Q2_K = 10, Q3_K = 11, Q4_K = 12, Q5_K = 13, Q6_K = 14, Q8_K = 15,
            BF16 = 30, MXFP4 = 39, NVFP4 = 40;

    // ---- jam_status ----
    public static final int OK = 0, EINVAL = 1, EUNSUPPORTED = 2, EBUSY = 3;

    /** Resolve a knob from {@code -Dprop}, else its environment form ({@code jam.binding} → {@code JAM_BINDING}:
     *  upper-case, dots→underscores), else {@code def}. The env form matches the native {@code JAM_*} vars
     *  ({@code JAM_NUM_THREADS}, {@code JAM_ISA}), so the same setting works from a JVM launcher or the shell. */
    static String config(String prop, String def) {
        String v = System.getProperty(prop);
        if (v == null || v.isEmpty()) v = System.getenv(prop.toUpperCase(Locale.ROOT).replace('.', '_'));
        return (v == null || v.isEmpty()) ? def : v;
    }

    /** Backend selector: JNI by default (marginally faster + proven); {@code -Djam.binding=ffm} or
     *  {@code JAM_BINDING=ffm} opts into the Panama path. {@code static final} so the JIT constant-folds
     *  the branch in {@link #mmUnsafe}. */
    private static final boolean FFM = "ffm".equalsIgnoreCase(config("jam.binding", "jni"));

    /** Panama downcall to {@code jam_mm}. Always linked (cheap, once), so {@link #mmFfm} is usable regardless
     *  of {@link #FFM} — the selector only picks the DEFAULT backend for {@link #mmUnsafe}. */
    private static final MethodHandle MM_FFM;
    static {
        NativeLoader.load();   // extracts & loads the bundled libjam (or the -Djam.library.path override)
        MM_FFM = Linker.nativeLinker().downcallHandle(
                SymbolLookup.loaderLookup().find("jam_mm").orElseThrow(
                    () -> new UnsatisfiedLinkError("jam: exported symbol 'jam_mm' not found")),
                // jam_mm(ctx, w,wt,ldw, a,at,lda, c,ct,ldc, m,n,k) — pointers as raw 64-bit addresses
                FunctionDescriptor.of(JAVA_INT,
                    JAVA_LONG,                     // jam_ctx* ctx (0 = global)
                    JAVA_LONG, JAVA_INT, JAVA_INT, // w, wt, ldw
                    JAVA_LONG, JAVA_INT, JAVA_INT, // a, at, lda
                    JAVA_LONG, JAVA_INT, JAVA_INT, // c, ct, ldc
                    JAVA_INT, JAVA_INT, JAVA_INT)); // m, n, k
    }

    /** JNI backend ({@code jam_jni.c}). Raw addresses; caller-managed liveness. */
    static native int mmJni(long a, int at, int lda,
                            long b, int bt, int ldb,
                            long c, int ct, int ldc,
                            int m, int n, int k);

    /** Panama backend: downcall straight to {@code jam_mm} on the global context. */
    static int mmFfm(long a, int at, int lda,
                     long b, int bt, int ldb,
                     long c, int ct, int ldc,
                     int m, int n, int k) {
        try {
            return (int) MM_FFM.invokeExact(0L,   // global ctx
                    a, at, lda, b, bt, ldb, c, ct, ldc, m, n, k);
        } catch (Throwable t) {
            throw new AssertionError("unreachable: jam_mm", t);
        }
    }

    /**
     * Package-private raw matmul: A/B/C are off-heap addresses with NO liveness or bounds checks — the
     * caller owns both. {@link #mm} routes through this after its checks; the white-box tests use it to drive
     * the raw / error paths. NOT public — the public surface is the bounds-checked {@link #mm} only.
     */
    static int mmUnsafe(long a, int at, int lda,
                        long b, int bt, int ldb,
                        long c, int ct, int ldc,
                        int m, int n, int k) {
        return FFM ? mmFfm(a, at, lda, b, bt, ldb, c, ct, ldc, m, n, k)
                   : mmJni(a, at, lda, b, bt, ldb, c, ct, ldc, m, n, k);
    }

    /**
     * The matmul: {@code C = A @ Bᵀ} on the global context (the sole public entry point). Each operand is a
     * {@link MemorySegment} plus a BYTE offset into it; the segments are kept reachable across the native call
     * (no premature free) and the call allocates nothing. Strides are ELEMENT row strides (k is unit-stride).
     * Returns a jam_status.
     *
     * <p>Each segment is BOUNDS-CHECKED against what the kernel will read/write, so a too-small segment throws
     * {@link IndexOutOfBoundsException} in Java instead of corrupting native memory. (Skipped for non-positive
     * dims/strides — native returns {@code EINVAL} — and for weight dtypes whose block layout isn't known here
     * — native returns {@code EUNSUPPORTED} without touching memory.)
     *
     * @param a,aOff  weight A [m×k] at {@code a.address()+aOff}, dtype {@code at}, element row stride {@code lda}
     * @param b,bOff  activations B [n×k], dtype {@code bt} (float), row stride {@code ldb}
     * @param c,cOff  output C [m×n], dtype {@code ct} (float), row stride {@code ldc}
     * @throws IndexOutOfBoundsException if any segment is too small for the requested matmul at its offset
     */
    public static int mm(MemorySegment a, long aOff, int at, int lda,
                         MemorySegment b, long bOff, int bt, int ldb,
                         MemorySegment c, long cOff, int ct, int ldc,
                         int m, int n, int k) {
        if (m > 0 && n > 0 && k > 0 && lda >= k && ldb >= k && ldc >= m) {   // else native classifies (EINVAL)
            checkSegment("weight A",     a, aOff, at, lda, m, k);   // [m×k] row-major, k elems/row at stride lda
            checkSegment("activation B", b, bOff, bt, ldb, n, k);   // [n×k] row-major
            checkSegment("output C",     c, cOff, ct, ldc, n, m);   // [m×n] token-major: n tokens × m features, stride ldc
        }
        try {
            return mmUnsafe(a.address() + aOff, at, lda,
                            b.address() + bOff, bt, ldb,
                            c.address() + cOff, ct, ldc, m, n, k);
        } finally {
            Reference.reachabilityFence(a);
            Reference.reachabilityFence(b);
            Reference.reachabilityFence(c);
        }
    }

    /** Verify {@code seg} holds the bytes the kernel touches for an operand of {@code nRows} rows ({@code rowElems}
     *  data each, at element row-stride {@code stride}) of dtype {@code dt}, starting at byte {@code off}. The
     *  element-stride → byte-span conversion (block-aware) lives in {@link GGMLType#spanBytes}. */
    private static void checkSegment(String which, MemorySegment seg, long off, int dt, int stride, int nRows, int rowElems) {
        GGMLType f = GGMLType.byCode(dt);
        if (f == null || !f.supported) return;       // unrecognized/unsupported -> native classifies; nothing to bound
        long need = f.spanBytes(nRows, stride, rowElems);
        if (off < 0 || off > seg.byteSize() - need)  // overflow-safe form of off + need > byteSize
            throw new IndexOutOfBoundsException(
                "jam.mm: " + which + " segment too small — need " + need + " B at offset " + off +
                ", segment is " + seg.byteSize() + " B");
    }

    private JAM() {}
}
