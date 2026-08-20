package com.qxotic.jinfer.telemetry;

import java.lang.System.Logger;
import java.lang.System.Logger.Level;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * The catalog of jinfer's silent performance cliffs: the situations where inference stays correct
 * but runs slower than it could, and would otherwise go unnoticed. Each constant is one cliff - it
 * carries its level and its full user-facing message, and reports at most once per JVM run (one log
 * line plus one {@link CliffEvent}).
 *
 * <p>Two rules keep this free where it matters:
 *
 * <ul>
 *   <li>Report ONLY from already-degraded paths. Healthy fast paths stay byte-identical; the plain
 *       read below is eliminable the one time it sits on a degraded path.
 *   <li>No per-call detail strings: the message lives on the constant, so a call site never
 *       allocates before the once-filter discards it. The cliff's location is statically known
 *       anyway - each constant has exactly one call site.
 * </ul>
 *
 * <p>Not everything slow is a cliff. Routine per-shape declines (flash-attention tails, dtype
 * oracles, conversion spans) are the dispatch working as designed and stay silent; a cliff is a
 * whole capability silently absent or degraded for the entire run.
 */
public enum PerformanceCliff {

    /** The active JIT compiles the incubating Vector API kernels conservatively. */
    SLOW_JIT(
            Level.WARNING,
            "this JVM's JIT compiles the incubating Vector API kernels conservatively - correct,"
                    + " but below their full potential for quantized decode and prefill; please"
                    + " upgrade to the latest Oracle GraalVM's JIT for better performance"),

    /** No native access, so every vector access keeps its bounds/liveness checks. */
    NATIVE_ACCESS_RESTRICTED(
            Level.WARNING,
            "native access is restricted, so the vector kernels keep per-access checks they"
                    + " could otherwise skip; --enable-native-access=ALL-UNNAMED lifts that small"
                    + " but uniform tax"),

    /** No jam backend on the classpath: prefill runs on the pure-Java path. */
    JAM_ABSENT(
            Level.WARNING,
            "no jam backend was found on the classpath, so prefill runs on the pure-Java path;"
                    + " com.qxotic:jam-native (or jam-vector) enables the fast kernels"),

    /** A second concurrent decode lost the spin pool and runs at roughly half bandwidth. */
    DECODE_CONTENTION(
            Level.WARNING,
            "a second decode is sharing this process and runs at roughly half decode bandwidth -"
                    + " expected when serving parallel sessions; nothing to fix unless"
                    + " single-stream latency matters more"),

    /** A jam backend declined a shape it was offered (EINVAL/EBUSY; smells like a bug). */
    JAM_DECLINE(
            Level.WARNING,
            "the native kernels declined a matmul shape they were offered, so that shape uses"
                    + " the Java path; if this persists across runs, we'd appreciate a report"),

    /** This model's mamba2 scan geometry isn't covered by the vector kernels. */
    MAMBA2_SCALAR(
            Level.WARNING,
            "this model's mamba2 scan geometry isn't covered by the vector kernels yet, so its"
                    + " recurrent scan - the model's hot loop - runs scalar"),

    /** This model's gated-delta scan geometry isn't covered by the vector kernels. */
    GDN_SCALAR(
            Level.WARNING,
            "this model's gated-delta scan geometry isn't covered by the vector kernels yet,"
                    + " so its recurrent scan - the model's hot loop - runs scalar"),

    /** No checkpoint codec for this model: every request prefills in full. */
    CACHE_SESSIONS_ONLY(
            Level.INFO,
            "this model has no checkpoint codec, so prompts are prefilled in full on every"
                    + " request; expected for this model, nothing to fix");

    private static final Logger LOG = System.getLogger("jinfer.perf");

    private final Level level;
    private final String message;
    private final AtomicBoolean reported = new AtomicBoolean();

    PerformanceCliff(Level level, String message) {
        this.level = level;
        this.message = message;
    }

    /**
     * Reports this cliff once per JVM run: one {@link CliffEvent} and one log line. Later calls
     * cost one eliminable plain read. Safe to call from any thread.
     */
    public void report() {
        if (!reported.getPlain() && reported.compareAndSet(false, true)) {
            CliffEvent.emit(name(), message);
            LOG.log(level, "perf cliff [{0}]: {1} (reported once per JVM run)", name(), message);
        }
    }
}
