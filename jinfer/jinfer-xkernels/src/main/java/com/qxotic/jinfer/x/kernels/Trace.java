package com.qxotic.jinfer.x.kernels;

import static com.qxotic.jinfer.x.Segments.readFloat;

import com.qxotic.jinfer.x.Views;
import com.qxotic.jinfer.x.Views.Raw;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;

/**
 * Per-layer residual-checkpoint tracing for the model ports, enabled with {@code -Djinfer.trace}.
 * Zero cost in production: {@link #sum} returns immediately when disabled, and callers guard the
 * expensive argument construction on {@link #ENABLED}. Ported from jinfer-kernels {@code Trace}
 * with the FloatTensor reads replaced by FP32-checked raw reads.
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
    public static void sum(String name, MemoryView<MemorySegment> t, int n) {
        if (!ENABLED) return;
        Raw r = Views.rawF32(t, "trace");
        double s = 0;
        for (int i = 0; i < n; i++) s += readFloat(r.vseg(), r.vbase() + (long) i * Float.BYTES);
        System.err.printf(
                "[trace] %s sum=%.6f v0=%.4f,%.4f,%.4f%n",
                name,
                s,
                readFloat(r.vseg(), r.vbase()),
                readFloat(r.vseg(), r.vbase() + Float.BYTES),
                readFloat(r.vseg(), r.vbase() + 2L * Float.BYTES));
    }
}
