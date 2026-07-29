package com.qxotic.jinfer.spring.ai.autoconfigure;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.context.properties.bind.DefaultValue;

/**
 * Configuration properties for jinfer reranking, bound under {@code spring.ai.jinfer.rerank}
 * (constructor binding).
 *
 * @param modelPath path to the reranker GGUF (e.g. a Qwen3-Reranker model); configuring it is what
 *     activates the post-processor
 * @param contextLength bounds query+document length (default 2048); {@code <= 0} = the model's own
 *     maximum
 * @param instruction the task instruction in the judge frame; empty = the model card's own wording
 * @param topK keep only the best {@code topK} documents; 0 (default) keeps all of them
 * @param minScore drop documents scoring below this; 0 (default) keeps all. The verdict is a
 *     probability, so 0.5 reads as "the model would have answered yes"
 */
@ConfigurationProperties("spring.ai.jinfer.rerank")
public record JinferRerankProperties(
        String modelPath,
        @DefaultValue("2048") int contextLength,
        String instruction,
        @DefaultValue("0") int topK,
        @DefaultValue("0") double minScore) {}
