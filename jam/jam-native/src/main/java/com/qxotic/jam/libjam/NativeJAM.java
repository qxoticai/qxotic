package com.qxotic.jam.libjam;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import com.qxotic.jam.JAM;
import com.qxotic.jam.internal.MemoryChecks;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.lang.ref.Reference;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * The native ({@code libjam}) {@link JAM} implementation - a handle to a jam context (a {@code
 * jam_context*}) driven by one host {@link JAM.Parallel}: {@link #create} sizes the context's
 * per-slot scratch by the pool's width and every fan-out comes back through {@link #pfUpcall} onto
 * that pool. libjam creates no thread inside a JVM.
 *
 * <p>{@link #mm} rejects heap segments, bounds-checks each native {@link MemorySegment} against
 * what the kernel touches, and keeps them reachable across the native call. One native {@code
 * jam_mm} is reached via the JNI shim (default) or a Panama downcall, selected once via {@code
 * -Djam.native.binding} (or {@code JAM_NATIVE_BINDING}).
 *
 * <p>{@code -Djam.native.isa} (or {@code JAM_NATIVE_ISA}) caps the kernel level by its jam.h name
 * ({@code generic}, {@code avx2}, {@code i8mm}, ...); unset, the best the machine has, the Apple
 * GPU included.
 *
 * <p><b>Concurrency:</b> a context is a single serial stream - concurrent calls on one context
 * serialize through a fair (FIFO) lock, so each waits its turn. {@code -Djam.native.serial=false}
 * (or {@code JAM_NATIVE_SERIAL=false}) restores the raw behavior where a contended call surfaces
 * {@link JAM#EBUSY} from the native guard (callers typically fall back to another backend). In
 * either mode {@link #close} waits for the call in flight before the context goes.
 */
public final class NativeJAM implements JAM, AutoCloseable {

    // jam.h's C types as FFM layouts: pointers and sizes cross as 64-bit values (jam.c asserts it).
    private static final ValueLayout C_INT32 = JAVA_INT;
    private static final ValueLayout C_UINT32 = JAVA_INT;
    private static final ValueLayout C_INT64 = JAVA_LONG;
    private static final ValueLayout C_POINTER = JAVA_LONG;

    private volatile long ctx; // owned jam_context*; 0 once closed
    private final JAM.Parallel parallel; // the host pool every fan-out of this context runs on

    /** Instances by the {@code user} value jam passes back to the upcall (their index). */
    private static final List<NativeJAM> INSTANCES = new CopyOnWriteArrayList<>();

    /**
     * {@code -Djam.native.serial} (or {@code JAM_NATIVE_SERIAL}): default {@code true} - concurrent
     * mm calls on one context serialize through a fair (FIFO) lock, so contended callers wait their
     * turn; {@code false} restores the raw behavior where a contended call surfaces {@code EBUSY}
     * from the native guard and callers fall back.
     */
    private static final boolean SERIAL =
            Boolean.parseBoolean(NativeLoader.config("jam.native.serial", "true"));

    /**
     * {@code -Djam.native.metal.resident}: default {@code true} - on macOS 15 and later, memory
     * shared with Metal stays wired while shared, so a GPU matmul starts at once; {@code false}
     * leaves it pageable, and every call then waits about 45 us per MB of the buffers it binds.
     */
    private static final boolean METAL_RESIDENT =
            Boolean.parseBoolean(System.getProperty("jam.native.metal.resident", "true"));

    // One fair lock per instance: close() and, when SERIAL, mm hold the write side; otherwise mm
    // holds the read side and the native guard arbitrates. So no call overlaps a close, and a
    // fan-out callback always resolves the instance that started it - never a cleared slot, never
    // a successor that reused it with another width.
    private final ReentrantReadWriteLock lock = new ReentrantReadWriteLock(true);
    private final Lock callLock = SERIAL ? lock.writeLock() : lock.readLock();

    private NativeJAM(long ctx, JAM.Parallel parallel) {
        this.ctx = ctx;
        this.parallel = parallel;
    }

    /**
     * A context whose fan-outs run on {@code parallel}, with per-slot scratch for its width. It
     * holds a native context until {@link #close()}.
     *
     * @throws IllegalArgumentException for a {@code jam.native.isa} that names no level
     * @throws IllegalStateException when jam refuses the context, with jam's reason
     */
    public static synchronized NativeJAM create(JAM.Parallel parallel) {
        int maxIsa = isa(NativeLoader.config("jam.native.isa", ""));
        int id = INSTANCES.indexOf(null); // a closed instance's handle is reused
        if (id < 0) id = INSTANCES.size();
        long ctx = newJni(parallel.width(), maxIsa, PF_STUB.address(), id, METAL_RESIDENT);
        if (ctx == 0) throw new IllegalStateException("jam: " + lastError());
        NativeJAM jam = new NativeJAM(ctx, parallel);
        if (id == INSTANCES.size()) INSTANCES.add(jam);
        else INSTANCES.set(id, jam);
        return jam;
    }

    /** Frees the native context once no call is in flight; later calls throw. Idempotent. */
    @Override
    public void close() {
        lock.writeLock().lock(); // after every mm and packSize in flight
        try {
            long handle;
            synchronized (NativeJAM.class) {
                handle = ctx;
                if (handle == 0) return;
                ctx = 0;
                INSTANCES.set(INSTANCES.indexOf(this), null);
            }
            freeJni(handle);
        } finally {
            lock.writeLock().unlock();
        }
    }

    long ctx() { // package-private: tests
        long c = ctx;
        if (c == 0) throw new IllegalStateException("jam: context is closed");
        return c;
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
        MemoryChecks.requireNative(w, "weight W");
        MemoryChecks.requireNative(a, "activation A");
        MemoryChecks.requireNative(r, "result R");
        if (m > 0 && n > 0 && k > 0 && ldw >= k && lda >= k
                && ldr >= m) { // else native classifies (EINVAL)
            if ((wt & PACKED) != 0) {
                // the packed layout is one blob of packSize bytes; native reads it only when that
                // size is non-zero (otherwise EUNSUPPORTED, no bytes touched)
                MemoryChecks.checkSpan("weight W", w, wOff, packSize(wt & ~PACKED, m, k));
            } else {
                MemoryChecks.checkSegment(
                        "weight W",
                        w,
                        wOff,
                        wt,
                        ldw,
                        m,
                        k); // [m×k] row-major, k elems/row at stride ldw
            }
            MemoryChecks.checkSegment("activation A", a, aOff, at, lda, n, k); // [n×k] row-major
            MemoryChecks.checkSegment(
                    "result R", r, rOff, rt, ldr, n, m); // [m×n] token-major: n tokens × m features
        }
        long wa = w.address() + wOff, aa = a.address() + aOff, ra = r.address() + rOff;
        callLock.lock();
        try {
            long ctx = ctx();
            int status =
                    FFM
                            ? mmFfm(ctx, wa, wt, ldw, aa, at, lda, ra, rt, ldr, m, n, k)
                            : mmJni(ctx, wa, wt, ldw, aa, at, lda, ra, rt, ldr, m, n, k);
            Throwable failed = UPCALL_FAILURE.get();
            if (failed != null) { // a task threw inside a fan-out: the result is garbage
                UPCALL_FAILURE.remove();
                if (failed instanceof RuntimeException e) throw e;
                if (failed instanceof Error e) throw e;
                throw new RuntimeException(failed);
            }
            return status;
        } finally {
            callLock.unlock();
            Reference.reachabilityFence(w);
            Reference.reachabilityFence(a);
            Reference.reachabilityFence(r);
        }
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

    /**
     * A host-driven context: {@code parallel} is an upcall stub with the {@code jam_parallel_fn}
     * signature, {@code user} the value jam passes back to it, {@code threads} how many slot values
     * the host will pass, {@code maxIsa} the cap. 0 on failure, the reason in {@link #lastError}.
     */
    private static native long newJni(
            int threads, int maxIsa, long parallel, long user, boolean metalResident);

    private static native void freeJni(long ctx);

    /** The calling thread's {@code jam_last_error} message. */
    static String lastError() {
        try {
            long message = (long) LAST_ERROR_FFM.invokeExact(0L); // the code: not asked for
            return MemorySegment.ofAddress(message).reinterpret(Long.MAX_VALUE).getString(0);
        } catch (Throwable t) {
            throw new AssertionError("unreachable: jam_last_error", t);
        }
    }

    /**
     * A jam.h {@code JAM_ISA_*} value by its {@code jam_isa_name}, asked of the library itself; ""
     * and "auto" are no cap.
     */
    static int isa(String name) {
        String wanted = name.toLowerCase(Locale.ROOT);
        if (wanted.isEmpty()) return 0;
        List<String> known = new ArrayList<>();
        for (int level = 0; level < 32; level++) { // jam.h's levels sit below 32
            String levelName = isaName(level);
            if (levelName.equals(wanted)) return level;
            if (!levelName.isEmpty()) known.add(levelName);
        }
        throw new IllegalArgumentException(
                "jam.native.isa=" + name + " names no level: " + String.join(", ", known));
    }

    private static String isaName(int level) {
        try {
            long name = (long) ISA_NAME_FFM.invokeExact(level);
            return MemorySegment.ofAddress(name).reinterpret(Long.MAX_VALUE).getString(0);
        } catch (Throwable t) {
            throw new AssertionError("unreachable: jam_isa_name", t);
        }
    }

    /**
     * {@code jam_pack_size} through Panama (a load-time call, so no JNI twin). Guarded by the pack
     * ABI: the Java packer ({@code JamPack}) is written against jam.h {@code JAM_PACK_ABI} 1, so a
     * library reading a different layout generation gets canonical weights instead.
     */
    private static final int PACK_ABI = 1;

    @Override
    public long packSize(int dtype, int m, int k) {
        if (PACK_ABI_NATIVE != PACK_ABI) return 0;
        lock.readLock().lock();
        try {
            long ctx = ctx();
            try {
                return (long) PACK_SIZE_FFM.invokeExact(ctx, dtype, m, k);
            } catch (Throwable t) {
                throw new AssertionError("unreachable: jam_pack_size", t);
            }
        } finally {
            lock.readLock().unlock();
        }
    }

    /**
     * {@code jam_share} with the release hung on the segment's scope: {@link #shareScope} lends
     * that scope to {@code reinterpret}, whose cleanup runs {@code jam_unshare} when the scope
     * closes. The cleanup captures only this instance and the address - capturing the segment would
     * keep an automatic arena alive forever.
     */
    @Override
    public boolean share(MemorySegment segment, boolean immutable) {
        if (!segment.isNative()) return false;
        // the same answer on every platform, not only where reinterpret runs below
        if (!segment.scope().isAlive()) throw new IllegalStateException("Already closed");
        if (!segment.isAccessibleBy(Thread.currentThread()))
            throw new WrongThreadException("Attempted access outside owning thread");
        if (segment.byteSize() == 0) return false;
        long addr = segment.address(), size = segment.byteSize();
        lock.readLock().lock();
        try {
            long ctx = ctx();
            if (jamShare(ctx, addr, size, immutable ? SHARE_IMMUTABLE : 0) != OK) return false;
            try {
                MemorySegment.ofAddress(addr)
                        .reinterpret(size, shareScope(segment.scope()), s -> unshare(addr));
            } catch (RuntimeException | Error e) { // closed meanwhile: no share left behind
                jamUnshare(ctx, addr);
                throw e;
            }
            return true;
        } finally {
            lock.readLock().unlock();
            Reference.reachabilityFence(segment);
        }
    }

    /** The scope-close end of a share; after {@link #close} the context already released it. */
    private void unshare(long addr) {
        lock.readLock().lock();
        try {
            long c = ctx;
            if (c != 0) jamUnshare(c, addr);
        } finally {
            lock.readLock().unlock();
        }
    }

    /**
     * An arena that only names {@code scope}: {@code reinterpret} reads nothing but {@link
     * Arena#scope()}, so this registers a cleanup on a scope whose arena the caller keeps.
     */
    private static Arena shareScope(MemorySegment.Scope scope) {
        return new Arena() {
            @Override
            public MemorySegment.Scope scope() {
                return scope;
            }

            @Override
            public MemorySegment allocate(long byteSize, long byteAlignment) {
                throw new UnsupportedOperationException("a view of another arena's scope");
            }

            @Override
            public void close() {
                throw new UnsupportedOperationException("a view of another arena's scope");
            }
        };
    }

    /** jam.h {@code JAM_SHARE_IMMUTABLE}. */
    private static final int SHARE_IMMUTABLE = 0x1;

    static int jamShare(long ctx, long addr, long size, int flags) { // package-private: tests
        try {
            return (int) SHARE_FFM.invokeExact(ctx, addr, size, flags);
        } catch (Throwable t) {
            throw new AssertionError("unreachable: jam_share", t);
        }
    }

    private static void jamUnshare(long ctx, long addr) {
        try {
            UNSHARE_FFM.invokeExact(ctx, addr);
        } catch (Throwable t) {
            throw new AssertionError("unreachable: jam_unshare", t);
        }
    }

    // ── the host's pool: jam fans its row ranges through it, never through native workers. ──

    /** A throwable from a task inside the current {@link #mm}; the upcall cannot throw. */
    private static final ThreadLocal<Throwable> UPCALL_FAILURE = new ThreadLocal<>();

    // jam_task_fn: void task(int32_t begin, int32_t end, int32_t slot, void* task_user)
    private static final MethodHandle TASK_FFM =
            Linker.nativeLinker()
                    .downcallHandle(
                            FunctionDescriptor.ofVoid(C_INT32, C_INT32, C_INT32, C_POINTER));

    private static void runTask(long task, long taskUser, int begin, int end, int slot) {
        if (begin >= end) return;
        try {
            TASK_FFM.invokeExact(MemorySegment.ofAddress(task), begin, end, slot, taskUser);
        } catch (Throwable t) {
            throw new AssertionError("unreachable: jam task", t);
        }
    }

    /**
     * The {@code jam_parallel_fn} upcall: [0,count) in at most {@code 4 x width} contiguous slices
     * on the instance's pool, each with its participant's slot. An exception cannot cross an upcall
     * stub (it would end the VM), so it is parked for {@link #mm} to rethrow once jam returns.
     */
    private static void pfUpcall(int count, long task, long taskUser, long user) {
        try {
            JAM.Parallel host = INSTANCES.get((int) user).parallel;
            int t = host.width();
            if (count < 2 || t < 2) {
                runTask(task, taskUser, 0, count, 0);
                return;
            }
            int slices = Math.min(count, 4 * t);
            int per = (count + slices - 1) / slices;
            host.run(
                    (count + per - 1) / per,
                    (s, slot) ->
                            runTask(task, taskUser, s * per, Math.min(count, s * per + per), slot));
        } catch (Throwable t) {
            if (UPCALL_FAILURE.get() == null) UPCALL_FAILURE.set(t);
        }
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
                                                int.class,
                                                long.class,
                                                long.class,
                                                long.class));
                // jam_parallel_fn: void parallel(int32_t count, jam_task_fn task, void* task_user,
                //                                void* user)
                stub =
                        Linker.nativeLinker()
                                .upcallStub(
                                        h,
                                        FunctionDescriptor.ofVoid(
                                                C_INT32, C_POINTER, C_POINTER, C_POINTER),
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

    /** Panama downcall to {@code jam_mm}: built only when the FFM backend is selected. */
    private static final MethodHandle MM_FFM;

    /**
     * Binds every native symbol this class needs and returns the library's {@code jam_pack_abi()}.
     * Class initialization does the binding, so a library that loaded but lacks a symbol (a stale
     * bundle) fails HERE, where {@link NativeJAMProvider#isAvailable()} can turn it into a
     * fallback, instead of on the first matmul.
     */
    static int probe() {
        return PACK_ABI_NATIVE;
    }

    /** {@code jam_pack_abi()} of the loaded library, read once. */
    private static final int PACK_ABI_NATIVE;

    private static final MethodHandle PACK_SIZE_FFM,
            SHARE_FFM,
            UNSHARE_FFM,
            LAST_ERROR_FFM,
            ISA_NAME_FFM;

    static {
        if (ADDRESS.byteSize() != 8
                || Linker.nativeLinker().canonicalLayouts().get("size_t").byteSize() != 8)
            throw new UnsatisfiedLinkError("jam: the native library needs a 64-bit platform");
        NativeLoader.load(); // always: the JNI backend needs libjam loaded too
        // jam_status jam_mm(jam_context*, const void* w, int32_t wt, int32_t ldw, const void* a,
        //                   int32_t at, int32_t lda, void* r, int32_t rt, int32_t ldr,
        //                   int32_t m, int32_t n, int32_t k)
        MM_FFM =
                !FFM
                        ? null
                        : bind(
                                "jam_mm",
                                FunctionDescriptor.of(
                                        C_INT32, C_POINTER, C_POINTER, C_INT32, C_INT32, C_POINTER,
                                        C_INT32, C_INT32, C_POINTER, C_INT32, C_INT32, C_INT32,
                                        C_INT32, C_INT32));
        // int32_t jam_pack_abi(void)
        MethodHandle abi = bind("jam_pack_abi", FunctionDescriptor.of(C_INT32));
        // int64_t jam_pack_size(const jam_context*, int32_t wt, int32_t m, int32_t k)
        PACK_SIZE_FFM =
                bind(
                        "jam_pack_size",
                        FunctionDescriptor.of(C_INT64, C_POINTER, C_INT32, C_INT32, C_INT32));
        // int32_t jam_share(jam_context*, const void* p, int64_t bytes, uint32_t flags)
        SHARE_FFM =
                bind(
                        "jam_share",
                        FunctionDescriptor.of(C_INT32, C_POINTER, C_POINTER, C_INT64, C_UINT32));
        // void jam_unshare(jam_context*, const void* p)
        UNSHARE_FFM = bind("jam_unshare", FunctionDescriptor.ofVoid(C_POINTER, C_POINTER));
        // const char* jam_last_error(int32_t* code)
        LAST_ERROR_FFM = bind("jam_last_error", FunctionDescriptor.of(C_POINTER, C_POINTER));
        // const char* jam_isa_name(int32_t isa)
        ISA_NAME_FFM = bind("jam_isa_name", FunctionDescriptor.of(C_POINTER, C_INT32));
        try {
            PACK_ABI_NATIVE = (int) abi.invokeExact();
        } catch (Throwable t) {
            throw new AssertionError("unreachable: jam_pack_abi", t);
        }
    }

    private static MethodHandle bind(String name, FunctionDescriptor descriptor) {
        MemorySegment symbol =
                SymbolLookup.loaderLookup()
                        .find(name)
                        .orElseThrow(
                                () ->
                                        new UnsatisfiedLinkError(
                                                "jam: the native library has no '"
                                                        + name
                                                        + "': it predates this jam-native"));
        return Linker.nativeLinker().downcallHandle(symbol, descriptor);
    }
}
