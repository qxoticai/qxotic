package com.qxotic.jinfer.telemetry;

import com.qxotic.jinfer.FloatTensor;
import com.qxotic.jinfer.RuntimeFlags;
import jdk.jfr.FlightRecorder;

/**
 * Registration for jinfer's sampled JFR events. Call {@link #install()} from anywhere a model is
 * loaded; it is idempotent, so every entry point can call it without coordinating.
 *
 * <p>There is deliberately no listener interface here: JFR IS the seam. It already carries
 * enable/disable, per-event settings and, through {@code RecordingStream}, in-process subscription
 * - so a Micrometer or OpenTelemetry exporter consumes the same events a user sees in JMC rather
 * than a second, parallel telemetry path.
 */
public final class Telemetry {

    private static volatile boolean installed;

    private Telemetry() {}

    /** Registers the periodic events. Idempotent and cheap; safe to call per model load. */
    public static synchronized void install() {
        if (installed) return;
        installed = true;
        FlightRecorder.addPeriodicEvent(
                RuntimeEvent.class,
                () -> {
                    RuntimeEvent event = new RuntimeEvent();
                    if (!event.isEnabled()) return;
                    event.vectorBits = FloatTensor.vectorBits();
                    event.decodeThreads = RuntimeFlags.DECODE_THREADS;
                    event.commit();
                });
    }
}
