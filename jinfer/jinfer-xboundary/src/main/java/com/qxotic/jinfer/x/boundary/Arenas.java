package com.qxotic.jinfer.x.boundary;

import java.lang.foreign.Arena;

/**
 * Arena flavors that adapt to the runtime. On the JVM {@link #newCrossThread()} is {@code
 * Arena.ofShared()} - closeable, freed deterministically. In a native image GraalVM's
 * SharedArenaSupport is mutually exclusive with the Vector API the kernels require (verified
 * through Oracle GraalVM 25.2.4), so it degrades to {@code Arena.ofAuto()}: cross-thread safe all
 * the same, but {@code close()} is a no-op there and the memory returns at GC. Owners already treat
 * close as best-effort on non-closeable arenas ({@link BaseState}), so the degrade is a
 * latency-of-free change, not a leak.
 *
 * <p>The image check runs at CALL time, not in a frozen static: a static final keyed off {@code
 * org.graalvm.nativeimage.imagecode} gets constant-folded when the class is build-time-initialized
 * (the property is not reliably visible to build-time class initialization), and the image then
 * bakes in {@code ofShared} arenas it cannot close.
 */
public final class Arenas {

    private Arenas() {}

    /** True on the JVM, and in a native image whose builder supports shared arenas. */
    public static boolean sharedArenas() {
        return System.getProperty("org.graalvm.nativeimage.imagecode") == null
                || Boolean.getBoolean("jinfer.sharedArenas");
    }

    /**
     * A cross-thread-safe arena, the best this runtime offers: {@code ofShared} on the JVM, {@code
     * ofAuto} in a native image. NOT named after a flavor - the returned arena may be neither
     * {@code ofShared} nor closeable, so callers must treat {@code close()} as best-effort (as
     * {@link BaseState} does).
     */
    public static Arena newCrossThread() {
        return sharedArenas() ? Arena.ofShared() : Arena.ofAuto();
    }
}
