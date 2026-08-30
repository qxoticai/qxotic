package com.qxotic.jinfer.telemetry;

import com.qxotic.jinfer.RuntimeFlags;
import com.qxotic.jinfer.Segments;
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
 *
 * <h2>Deliberate gaps</h2>
 *
 * <p>There is NO KV-cache gauge, though the obvious comparison (vLLM's cache-usage metric) has one.
 * That metric earns its place because paged attention allocates and frees KV blocks under pressure,
 * so occupancy genuinely fluctuates and predicts failure. jinfer pre-allocates each state's KV at
 * full context, once, so the same gauge would report two constants forever. The part that does vary
 * - running out of context - already surfaces per request as {@code finishReason=length}.
 *
 * <p>Consequently native memory is attributed only as far as {@link ModelLoadEvent}'s weights.
 * Per-state cost is not reported because nothing exposes a state's footprint. Allocation count is
 * reported by {@link PromptCacheEvent}; approximating its bytes would be less useful than admitting
 * they are unknown.
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
     * A registered engine's caches, sampled once a second. Deltas live here because a gauge needs
     * the previous reading to subtract from.
     *
     * <p>The gauge speaks {@link CacheSample} and {@link MediaCacheSample} - telemetry's own
     * vocabulary - so the event stream never names the cache modules' domain types, and this
     * package stays dependency-free.
     *
     * <p>LIFECYCLE, and it is the codebase's usual one - deterministic, with a GC backstop: {@link
     * #register} on construction, {@link #unregister} from {@code close()}. That is the contract,
     * and it is what a correct caller relies on.
     *
     * <p>The registry additionally holds this only WEAKLY, for one reason: an owner that is never
     * closed must stay collectable. Holding it strongly would pin that owner forever, and since
     * {@code LeakWatch} reports unclosed engines from a {@link java.lang.ref.Cleaner}, pinning them
     * would stop the leak detector firing - telemetry that silently disables jinfer's own leak
     * reporting. The weak reference is a backstop against that, never the mechanism.
     */
    public static final class CacheGauge {
        private final String model;
        private final Supplier<CacheSample> source;
        private final Supplier<MediaCacheSample> mediaSource;
        private long lastSessionHits, lastStateAllocations;
        private long lastBlockHits, lastBlockMisses, lastBlockEvictions;
        private long lastBlockDiscards, lastBlockRefusals;
        private long lastMediaHits, lastMediaMisses, lastMediaRefusals;

        public CacheGauge(
                String model,
                Supplier<CacheSample> source,
                Supplier<MediaCacheSample> mediaSource) {
            this.model = model;
            this.source = source;
            this.mediaSource = mediaSource;
        }

        /** Visible for tests. */
        String model() {
            return model;
        }

        private void emit() {
            PromptCacheEvent event = new PromptCacheEvent();
            if (!event.isEnabled()) return;
            CacheSample now = source.get();
            if (now == null) return;
            event.model = model;
            event.retainedSessions = now.retainedSessions();
            event.retainedSessionLimit = now.retainedSessionLimit();
            event.sessionHits = now.sessionHits() - lastSessionHits;
            event.stateAllocations = now.stateAllocations() - lastStateAllocations;
            event.sessionSnapshotBytes = now.sessionSnapshotBytes();
            event.blocks = now.blocks();
            event.bytes = now.bytes();
            event.budgetBytes = now.budgetBytes();
            event.blockHits = now.blockHits() - lastBlockHits;
            event.blockMisses = now.blockMisses() - lastBlockMisses;
            event.blockEvictions = now.blockEvictions() - lastBlockEvictions;
            event.blockDiscards = now.blockDiscards() - lastBlockDiscards;
            event.blockRefusals = now.blockRefusals() - lastBlockRefusals;
            lastSessionHits = now.sessionHits();
            lastStateAllocations = now.stateAllocations();
            lastBlockHits = now.blockHits();
            lastBlockMisses = now.blockMisses();
            lastBlockEvictions = now.blockEvictions();
            lastBlockDiscards = now.blockDiscards();
            lastBlockRefusals = now.blockRefusals();
            event.commit();
        }

        private void emitMedia() {
            MediaCacheEvent event = new MediaCacheEvent();
            if (!event.isEnabled()) return;
            MediaCacheSample now = mediaSource.get();
            if (now == null) return;
            event.model = model;
            event.entries = now.entries();
            event.bytes = now.bytes();
            event.budgetBytes = now.budgetBytes();
            event.hits = now.hits() - lastMediaHits;
            event.misses = now.misses() - lastMediaMisses;
            event.refusals = now.refusals() - lastMediaRefusals;
            lastMediaHits = now.hits();
            lastMediaMisses = now.misses();
            lastMediaRefusals = now.refusals();
            event.commit();
        }
    }

    private static final List<WeakReference<CacheGauge>> GAUGES = new CopyOnWriteArrayList<>();

    /**
     * Registers an engine's caches for sampling; pair it with {@link #unregister} in the owner's
     * close.
     *
     * <p>Prunes on the way in, because the periodic sweep only runs while a recording is active: a
     * process that never records and never closes its engines would otherwise accumulate one dead
     * reference per engine forever. Bounded by the number of live engines either way.
     */
    public static void register(CacheGauge gauge) {
        install();
        GAUGES.removeIf(reference -> reference.get() == null);
        GAUGES.add(new WeakReference<>(gauge));
    }

    /** Stops sampling {@code gauge}. Idempotent, like every close in this codebase. */
    public static void unregister(CacheGauge gauge) {
        GAUGES.removeIf(
                reference -> {
                    CacheGauge registered = reference.get();
                    return registered == null || registered == gauge;
                });
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
                    event.vectorBits = Segments.vectorBits();
                    event.threads = RuntimeFlags.THREADS;
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
        FlightRecorder.addPeriodicEvent(
                MediaCacheEvent.class,
                () -> {
                    GAUGES.removeIf(reference -> reference.get() == null);
                    for (WeakReference<CacheGauge> reference : GAUGES) {
                        CacheGauge gauge = reference.get();
                        if (gauge != null) gauge.emitMedia();
                    }
                });
    }
}
