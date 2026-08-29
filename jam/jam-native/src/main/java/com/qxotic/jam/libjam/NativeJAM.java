package com.qxotic.jam.libjam;

import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import com.qxotic.jam.JAM;
import com.qxotic.jam.internal.GGMLType;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.lang.ref.Reference;
import java.util.concurrent.locks.ReentrantLock;

/**
 * The native ({@code libjam}) {@link JAM} implementation - a handle to a jam context (a {@code
 * jam_ctx*}; {@link #global()} is the process-wide Java context). The constructor is private and
 * the provider owns one native context for the life of the process.
 *
 * <p>{@link #mm} rejects heap segments, bounds-checks each native {@link MemorySegment} against
 * what the kernel touches, and keeps them reachable across the native call. One native {@code
 * jam_mm} is reached via the JNI shim (default) or a Panama downcall, selected once via {@code
 * -Djam.native.binding} (or {@code JAM_NATIVE_BINDING}).
 *
 * <p><b>Concurrency:</b> a context is a single serial stream - concurrent calls on one context
 * serialize through a fair (FIFO) lock, so each waits its turn. {@code -Djam.native.serial=false}
 * (or {@code JAM_NATIVE_SERIAL=false}) restores the raw behavior where a contended call surfaces
 * {@link JAM#EBUSY} from the native guard (callers typically fall back to another backend).
 */
public final class NativeJAM implements JAM {

    private final long ctx; // owned jam_ctx*

    /**
     * {@code -Djam.native.serial} (or {@code JAM_NATIVE_SERIAL}): default {@code true} - concurrent
     * mm calls on one context serialize through a fair (FIFO) lock, so contended callers wait their
     * turn; {@code false} restores the raw behavior where a contended call surfaces {@code EBUSY}
     * from the native guard and callers fall back. Declared before {@link #global()}'s
     * instantiation: the instance field {@link #mmLock} reads it during class init.
     */
    private static final boolean SERIAL =
            Boolean.parseBoolean(NativeLoader.config("jam.native.serial", "true"));

    // Non-null only when SERIAL: concurrent mm calls then serialize FIFO instead of bouncing off
    // the native serial-stream guard (EBUSY). Per-instance, so a future NativeJAM.create(...)
    // gets its own lock + ctx for free.
    private final ReentrantLock mmLock = SERIAL ? new ReentrantLock(true) : null;

    private NativeJAM(long ctx) {
        this.ctx = ctx;
    }

    private static final NativeJAM GLOBAL;

    /** The process-wide native context; its parallel work runs on the host's {@link #host}. */
    public static NativeJAM global() {
        return GLOBAL;
    }

    @Override
    public int mm(
            MemorySegment w,
            long wOff,
            int wt,
            int ldw,
            MemorySegment a,
            long aOff,
            int at,
            int lda,
            MemorySegment r,
            long rOff,
            int rt,
            int ldr,
            int m,
            int n,
            int k) {
        requireNative(w, "weight W"); // heap/array-backed segments have no usable native address
        requireNative(a, "activation A");
        requireNative(r, "result R");
        if (m > 0 && n > 0 && k > 0 && ldw >= k && lda >= k
                && ldr >= m) { // else native classifies (EINVAL)
            checkSegment(
                    "weight W",
                    w,
                    wOff,
                    wt,
                    ldw,
                    m,
                    k); // [m×k] row-major, k elems/row at stride ldw
            checkSegment("activation A", a, aOff, at, lda, n, k); // [n×k] row-major
            checkSegment(
                    "result R", r, rOff, rt, ldr, n, m); // [m×n] token-major: n tokens × m features
        }
        long wa = w.address() + wOff, aa = a.address() + aOff, ra = r.address() + rOff;
        if (SERIAL) mmLock.lock();
        try {
            return FFM
                    ? mmFfm(ctx, wa, wt, ldw, aa, at, lda, ra, rt, ldr, m, n, k)
                    : mmJni(ctx, wa, wt, ldw, aa, at, lda, ra, rt, ldr, m, n, k);
        } finally {
            if (SERIAL) mmLock.unlock();
            Reference.reachabilityFence(w);
            Reference.reachabilityFence(a);
            Reference.reachabilityFence(r);
        }
    }

    /**
     * A heap (array-backed) segment has no stable native address - its {@code address()} is a heap
     * offset, not a pointer, so the kernel would corrupt memory. Reject it before we ever call
     * native.
     */
    private static void requireNative(MemorySegment seg, String which) {
        if (!seg.isNative())
            throw new IllegalArgumentException(
                    "jam.mm: "
                            + which
                            + " must be a NATIVE (off-heap) MemorySegment - heap/array-backed has"
                            + " no native address");
    }

    /**
     * Verify {@code seg} holds the bytes the kernel touches for {@code nRows} rows of {@code
     * rowElems} elements (dtype {@code dt}) at element row-stride {@code stride}, starting at byte
     * {@code off}. The element-stride → byte-span conversion (block-aware) lives in {@link
     * GGMLType#spanBytes}.
     */
    private static void checkSegment(
            String which,
            MemorySegment seg,
            long off,
            int dt,
            int stride,
            int nRows,
            int rowElems) {
        GGMLType g = GGMLType.byCode(dt);
        if (g == null) return; // unrecognized/unsupported -> native classifies; nothing to bound
        long need = g.spanBytes(nRows, stride, rowElems);
        if (off < 0 || off > seg.byteSize() - need) // overflow-safe form of off + need > byteSize
        throw new IndexOutOfBoundsException(
                    "jam.mm: "
                            + which
                            + " segment too small - need "
                            + need
                            + " B at offset "
                            + off
                            + ", segment is "
                            + seg.byteSize()
                            + " B");
    }

    // ── backends: one native jam_mm, reached via JNI (default) or Panama. ──

    /** JNI binding ({@code jam_jni.c}). Raw addresses; caller-managed liveness. */
    private static native int mmJni(
            long ctx,
            long w,
            int wt,
            int ldw,
            long a,
            int at,
            int lda,
            long r,
            int rt,
            int ldr,
            int m,
            int n,
            int k);

    /** Create the process-wide Java context through jam's existing context API. */
    private static native long createJni(int threads);

    /**
     * As createJni with a host executor: pf is an upcall stub with the jam_parallel_for signature.
     */
    private static native long createPfJni(int threads, long pf);

    /**
     * {@code jam_pack_size} through Panama (a load-time call, so no JNI twin). Guarded by the pack
     * ABI: the Java packer ({@code JamPack}) is written against jam.h {@code JAM_PACK_ABI} 1, so a
     * library reading a different layout generation gets canonical weights instead.
     */
    private static final int PACK_ABI = 1;

    @Override
    public long packSize(int dtype, int m, int k) {
        if (PACK_ABI_NATIVE != PACK_ABI) return 0;
        try {
            return (long) PACK_SIZE_FFM.invokeExact(ctx, dtype, m, k);
        } catch (Throwable t) {
            throw new AssertionError("unreachable: jam_pack_size", t);
        }
    }

    // ── the host's pool: jam fans its row ranges through it, never through native workers. ──

    /** Set by the provider; the calling thread alone until then. */
    static volatile JAM.Parallel host = JAM.Parallel.INLINE;

    // generic task downcall: (fn, arg, begin, end, tid) -> void
    private static final MethodHandle TASK_FFM =
            Linker.nativeLinker()
                    .downcallHandle(
                            FunctionDescriptor.ofVoid(JAVA_LONG, JAVA_INT, JAVA_INT, JAVA_INT));

    private static void runTask(long fn, long arg, int begin, int end, int tid) {
        if (begin >= end) return;
        try {
            TASK_FFM.invokeExact(MemorySegment.ofAddress(fn), arg, begin, end, tid);
        } catch (Throwable t) {
            throw new AssertionError("unreachable: jam task", t);
        }
    }

    /** The jam_parallel_for upcall: split [0,n) across the host's pool, one tid per slice. */
    private static void pfUpcall(long pool, int n, long fn, long arg) {
        int t = host.width();
        if (n < 2 || t < 2) {
            runTask(fn, arg, 0, n, 0);
            return;
        }
        int per = (n + t - 1) / t;
        int slices = (n + per - 1) / per;
        host.forLoop(slices, s -> runTask(fn, arg, s * per, Math.min(n, s * per + per), s));
    }

    private static final MemorySegment PF_STUB;

    static {
        MemorySegment stub;
        {
            try {
                MethodHandle h =
                        MethodHandles.lookup()
                                .findStatic(
                                        NativeJAM.class,
                                        "pfUpcall",
                                        MethodType.methodType(
                                                void.class,
                                                long.class,
                                                int.class,
                                                long.class,
                                                long.class));
                stub =
                        Linker.nativeLinker()
                                .upcallStub(
                                        h,
                                        FunctionDescriptor.ofVoid(
                                                JAVA_LONG, JAVA_INT, JAVA_LONG, JAVA_LONG),
                                        Arena.global());
            } catch (ReflectiveOperationException e) {
                throw new AssertionError("unreachable: pfUpcall", e);
            }
        }
        PF_STUB = stub;
    }

    /** Panama binding: downcall straight to {@code jam_mm}. */
    private static int mmFfm(
            long ctx,
            long w,
            int wt,
            int ldw,
            long a,
            int at,
            int lda,
            long r,
            int rt,
            int ldr,
            int m,
            int n,
            int k) {
        try {
            return (int) MM_FFM.invokeExact(ctx, w, wt, ldw, a, at, lda, r, rt, ldr, m, n, k);
        } catch (Throwable t) {
            throw new AssertionError("unreachable: jam_mm", t);
        }
    }

    /**
     * {@code -Djam.native.binding} (or {@code JAM_NATIVE_BINDING}): {@code jni} (default, proven)
     * or {@code ffm} (Panama).
     */
    private static final boolean FFM =
            "ffm".equalsIgnoreCase(NativeLoader.config("jam.native.binding", "jni"));

    /**
     * Panama downcall to {@code jam_mm} - built only when the FFM backend is selected; {@code null}
     * under JNI.
     */
    private static final MethodHandle MM_FFM;

    static {
        NativeLoader.load(); // always: the JNI backend needs libjam loaded too
        long ctx = createPfJni(Runtime.getRuntime().availableProcessors(), PF_STUB.address());
        if (ctx == 0) throw new IllegalStateException("jam: failed to create native context");
        GLOBAL = new NativeJAM(ctx);
        MM_FFM =
                !FFM
                        ? null
                        : Linker.nativeLinker()
                                .downcallHandle(
                                        SymbolLookup.loaderLookup()
                                                .find("jam_mm")
                                                .orElseThrow(
                                                        () ->
                                                                new UnsatisfiedLinkError(
                                                                        "jam: exported symbol"
                                                                                + " 'jam_mm' not"
                                                                                + " found")),
                                        // jam_mm(ctx, w,wt,ldw, a,at,lda, r,rt,ldr, m,n,k)  -
                                        // pointers as raw 64-bit addresses
                                        FunctionDescriptor.of(
                                                JAVA_INT, JAVA_LONG, // jam_ctx* ctx
                                                JAVA_LONG, JAVA_INT, JAVA_INT, // w, wt, ldw
                                                JAVA_LONG, JAVA_INT, JAVA_INT, // a, at, lda
                                                JAVA_LONG, JAVA_INT, JAVA_INT, // r, rt, ldr
                                                JAVA_INT, JAVA_INT, JAVA_INT)); // m, n, k
    }

    /** {@code jam_pack_abi()} of the loaded library, read once. */
    private static final int PACK_ABI_NATIVE;

    /** {@code jam_pack_size(ctx, dtype, m, k)} downcall. */
    private static final MethodHandle PACK_SIZE_FFM;

    static {
        Linker linker = Linker.nativeLinker();
        SymbolLookup lookup = SymbolLookup.loaderLookup();
        MethodHandle abi =
                linker.downcallHandle(
                        lookup.find("jam_pack_abi")
                                .orElseThrow(
                                        () ->
                                                new UnsatisfiedLinkError(
                                                        "jam: 'jam_pack_abi' not found")),
                        FunctionDescriptor.of(JAVA_INT));
        PACK_SIZE_FFM =
                linker.downcallHandle(
                        lookup.find("jam_pack_size")
                                .orElseThrow(
                                        () ->
                                                new UnsatisfiedLinkError(
                                                        "jam: 'jam_pack_size' not found")),
                        FunctionDescriptor.of(JAVA_LONG, JAVA_LONG, JAVA_INT, JAVA_INT, JAVA_INT));
        try {
            PACK_ABI_NATIVE = (int) abi.invokeExact();
        } catch (Throwable t) {
            throw new AssertionError("unreachable: jam_pack_abi", t);
        }
    }
}
