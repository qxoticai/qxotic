package com.qxotic.jinfer.server;

/**
 * One server's observability: request/token counters and the Prometheus text exposition
 * (llama.cpp-style {@code /metrics}).
 *
 * <p>PER INSTANCE, because {@link Server#start} promises each call an independent instance with its
 * own worker queue and generation state - and these counters used to be static, so two servers in
 * one JVM reported each other's traffic and each other's uptime. Scrape one, see both.
 *
 * <p>Written by the single generation worker and read by the metrics handler, so plain volatiles
 * suffice: one writer, no compound updates across fields.
 */
final class Metrics {

    static final String CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8";

    private final long startNanos = System.nanoTime();
    private volatile long requests, promptTokens, completionTokens;
    private volatile long sessionPoolHits, cachedTokens;

    /**
     * Record one finished generation (called on the worker thread).
     *
     * <p>Prometheus counters ONLY. jinfer.Inference is emitted by ChatEngine, which the server now
     * generates through - this class emitted it too while the server had its own pass, and kept
     * emitting after the migration until a test caught two events per request. These five volatiles
     * stay because a scrape endpoint is a different consumer than a JFR recording.
     */
    void record(Reply reply) {
        requests++;
        promptTokens += reply.promptTokens();
        completionTokens += reply.completionTokens();
    }

    /**
     * Record one prompt-cache serve (worker thread): tier 1 = append-only on a pooled live session,
     * otherwise a tier-2 block restore; {@code restored} positions were reused.
     */
    void recordPromptCache(boolean tier1, int restored) {
        if (tier1) sessionPoolHits++;
        cachedTokens += restored;
    }

    /** Prometheus exposition: request/token totals, queue + worker gauges. */
    String exposition(Worker worker) {
        StringBuilder sb = new StringBuilder();
        metric(sb, "jinfer_uptime_seconds", "gauge", (System.nanoTime() - startNanos) / 1e9);
        metric(sb, "jinfer_requests_total", "counter", requests);
        metric(sb, "jinfer_prompt_tokens_total", "counter", promptTokens);
        metric(sb, "jinfer_completion_tokens_total", "counter", completionTokens);
        metric(sb, "jinfer_session_pool_hits_total", "counter", sessionPoolHits);
        metric(sb, "jinfer_cached_tokens_total", "counter", cachedTokens);
        metric(sb, "jinfer_queue_depth", "gauge", worker.queued());
        metric(sb, "jinfer_worker_busy", "gauge", worker.busy() ? 1 : 0);
        return sb.toString();
    }

    private static void metric(StringBuilder sb, String name, String type, Number value) {
        sb.append("# TYPE ")
                .append(name)
                .append(' ')
                .append(type)
                .append('\n')
                .append(name)
                .append(' ')
                .append(value)
                .append('\n');
    }
}
