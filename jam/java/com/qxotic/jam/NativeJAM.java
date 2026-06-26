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
 * The native ({@code libjam}) {@link JAM} implementation — a handle to a jam context (a {@code jam_ctx*};
 * {@link #global()} is the process-global context). The constructor is private and context creation isn't
 * bound, so the global handle is the only instance today; per-context use ({@code NativeJAM.create(...)})
 * is a non-breaking future add.
 *
 * <p>{@link #mm} rejects heap segments, bounds-checks each native {@link MemorySegment} against what the
 * kernel touches, and keeps them reachable across the native call. One native {@code jam_mm} is reached via
 * the JNI shim (default) or a Panama downcall, selected once via {@code -Djam.binding} (or {@code JAM_BINDING}).
 *
 * <p><b>Concurrency:</b> a context is a single serial stream — concurrent calls return {@code EBUSY}.
 */
public final class NativeJAM implements JAM {

    private final long ctx;                       // jam_ctx* (0 = global)
    private NativeJAM(long ctx) { this.ctx = ctx; }

    private static final NativeJAM GLOBAL = new NativeJAM(0L);

    /** The process-global, env-configured native context ({@code JAM_NUM_THREADS}, {@code JAM_ISA}). */
    public static NativeJAM global() { return GLOBAL; }

    @Override
    public int mm(MemorySegment w, long wOff, int wt, int ldw,
                  MemorySegment a, long aOff, int at, int lda,
                  MemorySegment r, long rOff, int rt, int ldr,
                  int m, int n, int k) {
        requireNative(w, "weight W");        // heap/array-backed segments have no usable native address
        requireNative(a, "activation A");
        requireNative(r, "result R");
        if (m > 0 && n > 0 && k > 0 && ldw >= k && lda >= k && ldr >= m) {   // else native classifies (EINVAL)
            checkSegment("weight W",     w, wOff, wt, ldw, m, k);   // [m×k] row-major, k elems/row at stride ldw
            checkSegment("activation A", a, aOff, at, lda, n, k);   // [n×k] row-major
            checkSegment("result R",     r, rOff, rt, ldr, n, m);   // [m×n] token-major: n tokens × m features
        }
        long wa = w.address() + wOff, aa = a.address() + aOff, ra = r.address() + rOff;
        try {
            return FFM ? mmFfm(ctx, wa, wt, ldw, aa, at, lda, ra, rt, ldr, m, n, k)
                       : mmJni(ctx, wa, wt, ldw, aa, at, lda, ra, rt, ldr, m, n, k);
        } finally {
            Reference.reachabilityFence(w);
            Reference.reachabilityFence(a);
            Reference.reachabilityFence(r);
        }
    }

    /** A heap (array-backed) segment has no stable native address — its {@code address()} is a heap offset,
     *  not a pointer, so the kernel would corrupt memory. Reject it before we ever call native. */
    private static void requireNative(MemorySegment seg, String which) {
        if (!seg.isNative())
            throw new IllegalArgumentException(
                "jam.mm: " + which + " must be a NATIVE (off-heap) MemorySegment — heap/array-backed has no native address");
    }

    /** Verify {@code seg} holds the bytes the kernel touches for {@code nRows} rows of {@code rowElems}
     *  elements (dtype {@code dt}) at element row-stride {@code stride}, starting at byte {@code off}. The
     *  element-stride → byte-span conversion (block-aware) lives in {@link GGMLType#spanBytes}. */
    private static void checkSegment(String which, MemorySegment seg, long off, int dt, int stride, int nRows, int rowElems) {
        GGMLType g = GGMLType.byCode(dt);
        if (g == null) return;                          // unrecognized/unsupported -> native classifies; nothing to bound
        long need = g.spanBytes(nRows, stride, rowElems);
        if (off < 0 || off > seg.byteSize() - need)     // overflow-safe form of off + need > byteSize
            throw new IndexOutOfBoundsException(
                "jam.mm: " + which + " segment too small — need " + need + " B at offset " + off +
                ", segment is " + seg.byteSize() + " B");
    }

    // ── backends: one native jam_mm, reached via JNI (default) or Panama. ctx is a jam_ctx* (0 = global). ──

    /** JNI binding ({@code jam_jni.c}). Raw addresses; caller-managed liveness. */
    private static native int mmJni(long ctx,
                                    long w, int wt, int ldw,
                                    long a, int at, int lda,
                                    long r, int rt, int ldr,
                                    int m, int n, int k);

    /** Panama binding: downcall straight to {@code jam_mm}. */
    private static int mmFfm(long ctx,
                             long w, int wt, int ldw,
                             long a, int at, int lda,
                             long r, int rt, int ldr,
                             int m, int n, int k) {
        try {
            return (int) MM_FFM.invokeExact(ctx, w, wt, ldw, a, at, lda, r, rt, ldr, m, n, k);
        } catch (Throwable t) {
            throw new AssertionError("unreachable: jam_mm", t);
        }
    }

    /** {@code -Djam.binding} (or {@code JAM_BINDING}): {@code jni} (default, proven) or {@code ffm} (Panama). */
    private static final boolean FFM = "ffm".equalsIgnoreCase(NativeLoader.config("jam.binding", "jni"));

    /** Panama downcall to {@code jam_mm} — built only when the FFM backend is selected; {@code null} under JNI. */
    private static final MethodHandle MM_FFM;
    static {
        NativeLoader.load();   // always: the JNI backend needs libjam loaded too
        MM_FFM = !FFM ? null : Linker.nativeLinker().downcallHandle(
                SymbolLookup.loaderLookup().find("jam_mm").orElseThrow(
                    () -> new UnsatisfiedLinkError("jam: exported symbol 'jam_mm' not found")),
                // jam_mm(ctx, w,wt,ldw, a,at,lda, r,rt,ldr, m,n,k) — pointers as raw 64-bit addresses
                FunctionDescriptor.of(JAVA_INT,
                    JAVA_LONG,                     // jam_ctx* ctx (0 = global)
                    JAVA_LONG, JAVA_INT, JAVA_INT, // w, wt, ldw
                    JAVA_LONG, JAVA_INT, JAVA_INT, // a, at, lda
                    JAVA_LONG, JAVA_INT, JAVA_INT, // r, rt, ldr
                    JAVA_INT, JAVA_INT, JAVA_INT));// m, n, k
    }
}
