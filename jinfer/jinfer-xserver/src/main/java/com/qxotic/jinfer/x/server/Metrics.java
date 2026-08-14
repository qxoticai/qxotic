package com.qxotic.jinfer.x.server;

/**
 * One server's observability: generation/token counters and the Prometheus text exposition
 * (llama.cpp-style {@code /metrics}).
 *
 * <p>PER INSTANCE, because {@link Server#start} promises each call an independent instance with its
 * own worker queue and generation state - and these counters used to be static, so two servers in
 * one JVM reported each other's traffic and each other's uptime. Scrape one, see both.
 *
 * <p>Generation totals are written by the worker; request outcomes can also be written by handler
 * threads, so updates and scrapes are synchronized.
 */
final class Metrics {

    enum Outcome {
        INVALID_REQUEST,
        REJECTED,
        CANCELLED,
        FAILED,
        CLIENT_DISCONNECTED
    }

    static final String CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8";

    private final long startNanos = System.nanoTime();
    private long completedGenerations, promptTokens, completionTokens;
    private long invalidRequests, rejectedRequests, cancelledRequests, failedRequests;
    private long clientDisconnects;
    private long sessionPoolHits, cachedTokens;
    private long speculationRequests, draftedTokens, acceptedTokens, verifyForwards;

    /**
     * Record one finished generation (called on the worker thread).
     *
     * <p>Prometheus counters ONLY. jinfer.Inference is emitted by ChatEngine, which the server now
     * generates through - this class emitted it too while the server had its own pass, and kept
     * emitting after the migration until a test caught two events per request. These counters stay
     * because a scrape endpoint is a different consumer than a JFR recording.
     */
    synchronized void record(Reply reply) {
        completedGenerations++;
        promptTokens += reply.promptTokens();
        completionTokens += reply.completionTokens();
        if (reply.speculation() != null) {
            speculationRequests++;
            draftedTokens += reply.speculation().drafted();
            acceptedTokens += reply.speculation().accepted();
            verifyForwards += reply.speculation().forwards();
        }
    }

    synchronized void record(Outcome outcome) {
        switch (outcome) {
            case INVALID_REQUEST -> invalidRequests++;
            case REJECTED -> rejectedRequests++;
            case CANCELLED -> cancelledRequests++;
            case FAILED -> failedRequests++;
            case CLIENT_DISCONNECTED -> clientDisconnects++;
        }
    }

    /**
     * Record one prompt-cache serve (worker thread): tier 1 = append-only on a pooled live session,
     * otherwise a tier-2 block restore; {@code restored} positions were reused.
     */
    synchronized void recordPromptCache(boolean tier1, int restored) {
        if (tier1) sessionPoolHits++;
        cachedTokens += restored;
    }

    /** Prometheus exposition: generation outcomes, token totals, queue + worker gauges. */
    synchronized String exposition(Worker worker) {
        StringBuilder sb = new StringBuilder();
        metric(sb, "jinfer_uptime_seconds", "gauge", (System.nanoTime() - startNanos) / 1e9);
        metric(sb, "jinfer_generations_completed_total", "counter", completedGenerations);
        metric(sb, "jinfer_generation_requests_invalid_total", "counter", invalidRequests);
        metric(sb, "jinfer_generation_requests_rejected_total", "counter", rejectedRequests);
        metric(sb, "jinfer_generation_requests_cancelled_total", "counter", cancelledRequests);
        metric(sb, "jinfer_generation_requests_failed_total", "counter", failedRequests);
        metric(sb, "jinfer_client_disconnects_total", "counter", clientDisconnects);
        metric(sb, "jinfer_prompt_tokens_total", "counter", promptTokens);
        metric(sb, "jinfer_completion_tokens_total", "counter", completionTokens);
        metric(sb, "jinfer_session_pool_hits_total", "counter", sessionPoolHits);
        metric(sb, "jinfer_cached_tokens_total", "counter", cachedTokens);
        metric(sb, "jinfer_speculation_requests_total", "counter", speculationRequests);
        metric(sb, "jinfer_speculation_drafted_tokens_total", "counter", draftedTokens);
        metric(sb, "jinfer_speculation_accepted_tokens_total", "counter", acceptedTokens);
        metric(sb, "jinfer_speculation_forwards_total", "counter", verifyForwards);
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
