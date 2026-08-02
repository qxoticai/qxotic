package com.qxotic.jinfer.server;

import com.qxotic.jinfer.*;
import com.qxotic.jinfer.llm.*;

/**
 * Server observability: lifetime request/token counters and the Prometheus text exposition
 * (llama.cpp-style {@code /metrics}). Counters are single-writer (the generation worker) and read
 * by the metrics handler, so plain volatiles suffice.
 */
final class Metrics {

    private Metrics() {}

    static final String CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8";

    private static final long START_NANOS = System.nanoTime();
    private static volatile long requests, promptTokens, completionTokens;
    private static volatile long sessionPoolHits, cachedTokens;

    /**
     * Record one finished generation (called on the worker thread).
     *
     * <p>Prometheus counters ONLY. jinfer.Inference is emitted by ChatEngine, which the server now
     * generates through - this class emitted it too while the server had its own pass, and kept
     * emitting after the migration until a test caught two events per request. These five volatiles
     * stay because a scrape endpoint is a different consumer than a JFR recording.
     */
    static void record(Reply reply) {
        requests++;
        promptTokens += reply.promptTokens();
        completionTokens += reply.completionTokens();
    }

    /**
     * Record one prompt-cache serve (worker thread): tier 1 = append-only on a pooled live session,
     * otherwise a tier-2 block restore; {@code restored} positions were reused.
     */
    static void recordPromptCache(boolean tier1, int restored) {
        if (tier1) sessionPoolHits++;
        cachedTokens += restored;
    }

    /** Prometheus exposition: request/token totals, queue + worker gauges. */
    static String exposition(Worker worker) {
        StringBuilder sb = new StringBuilder();
        metric(sb, "jinfer_uptime_seconds", "gauge", (System.nanoTime() - START_NANOS) / 1e9);
        metric(sb, "jinfer_requests_total", "counter", requests);
        metric(sb, "jinfer_prompt_tokens_total", "counter", promptTokens);
        metric(sb, "jinfer_completion_tokens_total", "counter", completionTokens);
        metric(sb, "jinfer_session_pool_hits_total", "counter", sessionPoolHits);
        metric(sb, "jinfer_cached_tokens_total", "counter", cachedTokens);
        metric(sb, "jinfer_queue_depth", "gauge", worker.queueDepth());
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
