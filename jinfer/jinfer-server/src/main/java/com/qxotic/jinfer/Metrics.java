package com.qxotic.jinfer;

import com.qxotic.jinfer.Engine.GenerationResult;

import java.util.Map;

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

    /** Record one finished generation (called on the worker thread). */
    static void record(GenerationResult result) {
        requests++;
        promptTokens += result.promptTokens();
        completionTokens += result.completionTokens();
    }

    /** Prometheus exposition: request/token totals, queue + worker gauges, prompt-cache stats. */
    static String exposition(Worker worker, PromptCache cache) {
        StringBuilder sb = new StringBuilder();
        metric(sb, "jinfer_uptime_seconds", "gauge", (System.nanoTime() - START_NANOS) / 1e9);
        metric(sb, "jinfer_requests_total", "counter", requests);
        metric(sb, "jinfer_prompt_tokens_total", "counter", promptTokens);
        metric(sb, "jinfer_completion_tokens_total", "counter", completionTokens);
        metric(sb, "jinfer_queue_depth", "gauge", worker.queueDepth());
        metric(sb, "jinfer_worker_busy", "gauge", worker.busy() ? 1 : 0);
        if (cache != null) {
            for (Map.Entry<String, Object> entry : cache.stats().entrySet()) {
                if (entry.getValue() instanceof Number n) {
                    String kind = entry.getKey().endsWith("bytes") || entry.getKey().equals("nodes") ? "gauge" : "counter";
                    metric(sb, "jinfer_prompt_cache_" + entry.getKey(), kind, n);
                }
            }
        }
        return sb.toString();
    }

    private static void metric(StringBuilder sb, String name, String type, Number value) {
        sb.append("# TYPE ").append(name).append(' ').append(type).append('\n')
          .append(name).append(' ').append(value).append('\n');
    }
}
