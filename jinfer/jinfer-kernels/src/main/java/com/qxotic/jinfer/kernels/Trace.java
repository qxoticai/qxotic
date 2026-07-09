package com.qxotic.jinfer.kernels;

import com.qxotic.jinfer.*;

/**
 * Per-layer residual-checkpoint tracing for the model ports, enabled with {@code -Djinfer.trace}.
 * Zero cost in production: {@link #sum} returns immediately when disabled, and callers guard the
 * expensive argument construction on {@link #ENABLED}.
 */
public final class Trace {
    private Trace() {}

    public static final boolean ENABLED = System.getProperty("jinfer.trace") != null;

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
