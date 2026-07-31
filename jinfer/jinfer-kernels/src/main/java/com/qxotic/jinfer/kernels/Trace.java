package com.qxotic.jinfer.kernels;

import com.qxotic.jinfer.*;

/**
 * Per-layer residual-checkpoint tracing for the model ports, enabled with {@code -Djinfer.trace}.
 * Zero cost in production: {@link #sum} returns immediately when disabled, and callers guard the
 * expensive argument construction on {@link #ENABLED}.
 */
public final class Trace {
    private Trace() {}

    /**
     * {@code -Djinfer.trace} (or {@code =false} to force off). Read at class init and deliberately
     * NOT run-time-initialized in a native image: its call sites sit inside per-layer loops, so a
     * folded constant erases the branch entirely, which a run-time flag could not. The cost is that
     * an image freezes this at BUILD time — pass it to the image build (the {@code jinfer.trace}
     * pom property) rather than to the binary, exactly as with {@code jinfer.convTile}.
     */
    public static final boolean ENABLED =
            !"false".equals(System.getProperty("jinfer.trace", "false"));

    /** Prints the span's sum and first three elements, tagged with {@code name}. */
    public static void sum(String name, FloatTensor t, int n) {
        if (!ENABLED) return;
        double s = 0;
        for (int i = 0; i < n; i++) s += t.getFloat(i);
        System.err.printf(
                "[trace] %s sum=%.6f v0=%.4f,%.4f,%.4f%n",
                name, s, t.getFloat(0), t.getFloat(1), t.getFloat(2));
    }
}
