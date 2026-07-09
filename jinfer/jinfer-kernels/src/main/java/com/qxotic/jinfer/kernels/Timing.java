package com.qxotic.jinfer.kernels;

import com.qxotic.jinfer.*;

/**
 * Bench-only nanosecond accumulators for attributing prefill cost to attention vs the rest. Zero
 * cost in production: callers only touch it when {@code -Djinfer.timing=true}.
 */
public final class Timing {
    public static final boolean ENABLED = Boolean.getBoolean("jinfer.timing");
    private static long attnNanos;

    public static void reset() {
        attnNanos = 0;
    }

    public static void addAttn(long nanos) {
        attnNanos += nanos;
    }

    public static double attnMs() {
        return attnNanos / 1e6;
    }

    private Timing() {}
}
