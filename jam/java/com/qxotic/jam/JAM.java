package com.qxotic.jam;

import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.lang.ref.Reference;

import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

/**
 * jam — fastest multithreaded CPU matmul: {@code C = A @ Bᵀ}, with quantized weights.
 *
 * <p>The Java surface runs on the <b>process-global</b> jam context, created lazily on first use and
 * configured through environment variables ({@code JAM_NUM_THREADS}, {@code JAM_ISA}); no handle to manage.
 *
 * <p>Two call mechanisms reach the same native {@code jam_mm}, selected once via {@code -Djam.binding}:
 * <ul>
 *   <li>{@code jni} (default) — the {@code jam_jni.c} shim ({@link #mmJni}); a hair faster, proven;</li>
 *   <li>{@code ffm} — a Panama downcall straight to the exported {@code jam_mm} (no shim to maintain).</li>
 * </ul>
 * The selector is a {@code static final} boolean, so the JIT folds it away — the inactive backend costs nothing.
 *
 * <p>Three entry points:
 * <ul>
 *   <li>{@link #mm(MemorySegment, long, int, int, MemorySegment, long, int, int, MemorySegment, long, int, int, int, int, int)}
 *       — the <b>safe</b> front door: segments + byte offsets, kept alive across the call, zero allocation;</li>
 *   <li>{@link #mmUnsafe} — raw off-heap addresses you keep alive yourself (e.g. a global Arena); the
 *       zero-overhead escape hatch routed to the configured backend;</li>
 *   <li>{@link #mmJni} — the JNI backend directly (mostly for A/B-ing the two binding overheads).</li>
 * </ul>
 *
 * <p><b>Concurrency:</b> the global context is a single serial execution stream — multithreaded <i>within</i>
 * a call, but calls must come from <b>one</b> thread (concurrent calls contend on the shared pool/scratch).
 */
public final class JAM {

    // ---- dtype tags: numerically identical to GGML's ggml_type (append-only ABI; never reorder) ----
    public static final int F32 = 0, F16 = 1,
            Q4_0 = 2, Q4_1 = 3, Q5_0 = 6, Q5_1 = 7, Q8_0 = 8,
            Q2_K = 10, Q3_K = 11, Q4_K = 12, Q5_K = 13, Q6_K = 14, Q8_K = 15,
            BF16 = 30, MXFP4 = 39;

    // ---- jam_status ----
    public static final int OK = 0, EINVAL = 1, EUNSUPPORTED = 2, EBUSY = 3;

    /** Backend selector: JNI by default (marginally faster + proven); {@code -Djam.binding=ffm} opts into
     *  the Panama path. {@code static final} so the JIT constant-folds the branch in {@link #mmUnsafe}. */
    private static final boolean FFM = "ffm".equalsIgnoreCase(System.getProperty("jam.binding", "jni"));

    /** Panama downcall to {@code jam_mm}; null when the JNI backend is active (so FFM is never set up). */
    private static final MethodHandle MM_FFM;
    static {
        NativeLoader.load();   // extracts & loads the bundled libjam (or the -Djam.library.path override)
        MM_FFM = FFM
                ? Linker.nativeLinker().downcallHandle(
                    SymbolLookup.loaderLookup().find("jam_mm").orElseThrow(
                        () -> new UnsatisfiedLinkError("jam: exported symbol 'jam_mm' not found")),
                    // jam_mm(ctx, w,wt,ldw, a,at,lda, c,ct,ldc, m,n,k) — pointers as raw 64-bit addresses
                    FunctionDescriptor.of(JAVA_INT,
                        JAVA_LONG,                     // jam_ctx* ctx (0 = global)
                        JAVA_LONG, JAVA_INT, JAVA_INT, // w, wt, ldw
                        JAVA_LONG, JAVA_INT, JAVA_INT, // a, at, lda
                        JAVA_LONG, JAVA_INT, JAVA_INT, // c, ct, ldc
                        JAVA_INT, JAVA_INT, JAVA_INT)) // m, n, k
                : null;
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
     * Raw matmul: A/B/C are off-heap addresses you are responsible for keeping alive for the call
     * (e.g. backed by a global Arena). No liveness or bounds checks — the zero-overhead escape hatch,
     * routed to the configured backend. Prefer {@link #mm} unless you own the operands' lifetime.
     */
    public static int mmUnsafe(long a, int at, int lda,
                               long b, int bt, int ldb,
                               long c, int ct, int ldc,
                               int m, int n, int k) {
        return FFM ? mmFfm(a, at, lda, b, bt, ldb, c, ct, ldc, m, n, k)
                   : mmJni(a, at, lda, b, bt, ldb, c, ct, ldc, m, n, k);
    }

    /**
     * Safe matmul: {@code C = A @ Bᵀ} on the global context. Each operand is a {@link MemorySegment} plus a
     * BYTE offset into it; the segments are kept reachable across the native call (no premature free) and the
     * call allocates nothing. Strides are ELEMENT row strides (k is unit-stride). Returns a jam_status.
     *
     * @param a,aOff  weight A [m×k] at {@code a.address()+aOff}, dtype {@code at}, element row stride {@code lda}
     * @param b,bOff  activations B [n×k], dtype {@code bt} (float), row stride {@code ldb}
     * @param c,cOff  output C [m×n], dtype {@code ct} (float), row stride {@code ldc}
     */
    public static int mm(MemorySegment a, long aOff, int at, int lda,
                         MemorySegment b, long bOff, int bt, int ldb,
                         MemorySegment c, long cOff, int ct, int ldc,
                         int m, int n, int k) {
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

    private JAM() {}
}
