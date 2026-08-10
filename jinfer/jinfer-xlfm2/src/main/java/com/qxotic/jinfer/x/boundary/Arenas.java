package com.qxotic.jinfer.x.boundary;

import java.lang.foreign.Arena;

/**
 * Arena flavors that adapt to the runtime. On the JVM {@link #newShared()} is {@code
 * Arena.ofShared()} - closeable, freed deterministically. In a native image GraalVM's
 * SharedArenaSupport is mutually exclusive with the Vector API the kernels require (verified
 * through Oracle GraalVM 25.2.4), so it degrades to {@code Arena.ofAuto()}: cross-thread safe all
 * the same, but {@code close()} is a no-op there and the memory returns at GC. Owners already treat
 * close as best-effort on non-closeable arenas ({@link BaseState}), so the degrade is a
 * latency-of-free change, not a leak.
 */
public final class Arenas {

    private Arenas() {}

    /** True on the JVM, and in a native image whose builder supports shared arenas. */
    public static final boolean SHARED_ARENAS =
            System.getProperty("org.graalvm.nativeimage.imagecode") == null
                    || Boolean.getBoolean("jinfer.sharedArenas");

    /** A cross-thread-safe arena: {@code ofShared} on the JVM, {@code ofAuto} in a native image. */
    public static Arena newShared() {
        return SHARED_ARENAS ? Arena.ofShared() : Arena.ofAuto();
    }
}
