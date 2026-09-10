package com.qxotic.jinfer.spring.ai.autoconfigure;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.context.properties.bind.DefaultValue;

/**
 * Configuration properties for jinfer reranking, bound under {@code spring.ai.jinfer.rerank}
 * (constructor binding).
 *
 * @param model the reranker GGUF (e.g. Qwen3-Reranker or LFM2.5-ColBERT) as a local path or model
 *     ref; configuring it is what activates the post-processor
 * @param contextCapacity upper bound on the encoded query-and-document context (default 2048); 0
 *     uses the model's maxContextLength; negative values are rejected
 * @param instruction the task instruction in the judge frame; empty = the model card's own wording
 * @param topK keep only the best topK documents; 0 (default) keeps all of them
 * @param minScore drop documents scoring below this; unset keeps all. Qwen3-Reranker scores are
 *     probabilities (0.5 reads as "the model would have answered yes"); LFM2.5-ColBERT scores are
 *     MaxSim SUMS (unbounded - rank and relative thresholds only)
 */
@ConfigurationProperties("spring.ai.jinfer.rerank")
public record JinferRerankProperties(
        String model,
        @DefaultValue("2048") int contextCapacity,
        String instruction,
        @DefaultValue("0") int topK,
        Double minScore) {}
