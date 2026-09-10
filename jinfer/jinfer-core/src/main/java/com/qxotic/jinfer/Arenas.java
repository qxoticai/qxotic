package com.qxotic.jinfer;

import com.qxotic.jota.memory.MemoryAllocators;
import com.qxotic.jota.memory.MemoryArena;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

/**
 * <b>WARNING: NEVER supply confined arenas for jinfer weights or state. Misuse can corrupt memory
 * or crash the JVM. This requirement is NOT enforced in normal execution.</b> Jinfer is
 * multi-threaded by design: even with one worker, execution may run on a custom pool's thread or a
 * native pthread other than the arena's owning Java thread. Reducing the worker count or
 * serializing calls does not make confined arenas supported. Raw-address and native kernels bypass
 * JDK confinement checks, so misuse may proceed without {@link WrongThreadException}. A successful
 * load or generation does not establish safety.
 *
 * <p>Enable {@code -ea:com.qxotic.jinfer...} (or {@code -ea}) for an assertion-only diagnostic at
 * weight-mapping and state-construction boundaries, once per mapping or state, never per token or
 * kernel call. It probes a zero-byte allocation against a private, never-started thread. The probe
 * and thread are created only with assertions enabled. This is a diagnostic, NOT a safety
 * guarantee: custom allocators may handle zero-byte allocations differently from real buffers, and
 * assertions may be disabled.
 *
 * <p>Use {@link #newCrossThread()}, {@link Arena#ofShared()}, {@link Arena#ofAuto()}, or {@link
 * Arena#global()}, and keep the arena alive until all borrowing operations have finished. Closing
 * borrowed memory during inference is also unsafe.
 *
 * <p>Arena flavors that adapt to the runtime. On the JVM {@link #newCrossThread()} is {@code
 * Arena.ofShared()} - closeable, freed deterministically. In a native image GraalVM's
 * SharedArenaSupport is mutually exclusive with the Vector API the kernels require (verified
 * through Oracle GraalVM 25.2.4), so it degrades to {@code Arena.ofAuto()}: cross-thread safe all
 * the same, but {@code close()} is a no-op there and the memory returns at GC. Owners already treat
 * close as best-effort on non-closeable arenas ({@link ContextState}), so the degrade is a
 * latency-of-free change, not a leak.
 *
 * <p>The image check runs at CALL time, not in a frozen static: a static final keyed off {@code
 * org.graalvm.nativeimage.imagecode} gets constant-folded when the class is build-time-initialized
 * (the property is not reliably visible to build-time class initialization), and the image then
 * bakes in {@code ofShared} arenas it cannot close.
 */
public final class Arenas {

    private Arenas() {}

    private static final String CONFINED_MESSAGE =
            "Confined arenas are unsupported by jinfer, even with one worker; use"
                + " Arenas.newCrossThread(), Arena.ofShared(), Arena.ofAuto(), or Arena.global(). A"
                + " custom pool or native pthread may access the memory and crash the JVM.";

    /** Assertion-only buffer-access diagnostic; see this class's memory-safety warning. */
    public static void assertCrossThread(Arena arena) {
        assert probeCrossThreadBufferAccess(MemoryAllocators.ofArena(arena)) : CONFINED_MESSAGE;
    }

    /**
     * Probes whether an allocated buffer permits cross-thread access, not whether allocation is
     * thread-safe. A zero-byte probe cannot certify a custom allocator's other buffers.
     */
    public static void assertCrossThread(MemoryArena<MemorySegment> arena) {
        assert probeCrossThreadBufferAccess(arena) : CONFINED_MESSAGE;
    }

    private static boolean probeCrossThreadBufferAccess(MemoryArena<MemorySegment> arena) {
        try {
            MemorySegment buffer = arena.allocateMemory(0, 1).base();
            // Local to the assertion: native images must not retain a Thread in a static field.
            Thread other =
                    Thread.ofPlatform().inheritInheritableThreadLocals(false).unstarted(() -> {});
            return buffer.isAccessibleBy(other);
        } catch (WrongThreadException failure) {
            return false;
        }
    }

    /** True on the JVM, and in a native image whose builder supports shared arenas. */
    public static boolean sharedArenas() {
        return System.getProperty("org.graalvm.nativeimage.imagecode") == null
                || Boolean.getBoolean("jinfer.sharedArenas");
    }

    /**
     * A cross-thread-safe arena, the best this runtime offers: {@code ofShared} on the JVM, {@code
     * ofAuto} in a native image. NOT named after a flavor - the returned arena may be neither
     * {@code ofShared} nor closeable, so callers must treat {@code close()} as best-effort (as
     * {@link ContextState} does).
     */
    public static Arena newCrossThread() {
        return sharedArenas() ? Arena.ofShared() : Arena.ofAuto();
    }

    /**
     * The default state arena for {@link ContextModel#newState(int, int, MemoryArena)}: Panama over
     * {@link #newCrossThread()}. Callers wanting GPU-shared state memory pass their own arena
     * instead - this factory is only the built-in default.
     */
    public static MemoryArena<MemorySegment> newCrossThreadMemoryArena() {
        return MemoryAllocators.adoptArena(newCrossThread());
    }

    /** Closes a runtime-selected arena when its implementation supports eager release. */
    public static void close(Arena arena) {
        if (!arena.scope().isAlive()) return;
        try {
            arena.close();
        } catch (UnsupportedOperationException ignored) {
            // ofAuto/global own their lifetime.
        }
    }

    /** Closes a memory arena when its implementation supports eager release. */
    public static void close(MemoryArena<?> arena) {
        if (!arena.isAlive()) return;
        try {
            arena.close();
        } catch (UnsupportedOperationException ignored) {
            // self-managing arenas own their lifetime.
        }
    }
}
