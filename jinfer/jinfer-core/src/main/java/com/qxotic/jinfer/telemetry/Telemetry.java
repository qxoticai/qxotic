package com.qxotic.jinfer.telemetry;

import com.qxotic.jinfer.FloatTensor;
import com.qxotic.jinfer.RuntimeFlags;
import com.qxotic.jinfer.cache.PromptCache;
import java.lang.ref.WeakReference;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.function.Supplier;
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

    /**
     * Queue wait handed from whoever owns a queue to whoever emits the event. A thread-local
     * because the job runs ON the worker thread that dequeued it, so the wait and the generation
     * share a thread but nothing else - and threading a nanosecond count through the whole request
     * API to serve one optional field would be worse.
     */
    private static final ThreadLocal<Long> QUEUE_WAIT = new ThreadLocal<>();

    private Telemetry() {}

    /** Records how long the current thread's job waited to be dequeued. */
    public static void queueWait(long nanos) {
        QUEUE_WAIT.set(nanos);
    }

    /** Reads and clears the pending queue wait; 0 when nothing queued this thread. */
    public static long takeQueueWait() {
        Long waited = QUEUE_WAIT.get();
        if (waited == null) return 0L;
        QUEUE_WAIT.remove();
        return waited;
    }

    /**
     * A registered prompt cache, sampled once a second. Deltas live here because a gauge needs the
     * previous reading to subtract from.
     *
     * <p>OWNERSHIP: the registry holds this WEAKLY, so whoever registers must keep it in a field
     * for as long as it should be sampled. Registering a lambda or a temporary would let the next
     * GC silently stop the gauge - and holding it strongly here would keep a dead engine's whole
     * cache alive forever, which is the failure this indirection exists to prevent.
     */
    public static final class CacheGauge {
        private final String model;
        private final Supplier<PromptCache.Sample> source;
        private long lastHits, lastMisses, lastEvictions;

        public CacheGauge(String model, Supplier<PromptCache.Sample> source) {
            this.model = model;
            this.source = source;
        }

        /** Visible for tests. */
        String model() {
            return model;
        }

        private void emit() {
            PromptCacheEvent event = new PromptCacheEvent();
            if (!event.isEnabled()) return;
            PromptCache.Sample now = source.get();
            if (now == null) return;
            event.model = model;
            event.blocks = now.blocks();
            event.bytes = now.bytes();
            event.budgetBytes = now.budgetBytes();
            event.hits = now.hits() - lastHits;
            event.misses = now.misses() - lastMisses;
            event.evictions = now.evictions() - lastEvictions;
            lastHits = now.hits();
            lastMisses = now.misses();
            lastEvictions = now.evictions();
            event.commit();
        }
    }

    private static final List<WeakReference<CacheGauge>> GAUGES = new CopyOnWriteArrayList<>();

    /** Registers a cache for sampling. Keep {@code gauge} in a field - see {@link CacheGauge}. */
    public static void register(CacheGauge gauge) {
        install();
        GAUGES.add(new WeakReference<>(gauge));
    }

    /** Visible for tests: how many gauges are still reachable, pruning the dead. */
    static int liveGauges() {
        GAUGES.removeIf(reference -> reference.get() == null);
        return GAUGES.size();
    }

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
        FlightRecorder.addPeriodicEvent(
                PromptCacheEvent.class,
                () -> {
                    // prune as we go: a collected gauge means its owner is gone
                    GAUGES.removeIf(reference -> reference.get() == null);
                    for (WeakReference<CacheGauge> reference : GAUGES) {
                        CacheGauge gauge = reference.get();
                        if (gauge != null) gauge.emit();
                    }
                });
    }
}
