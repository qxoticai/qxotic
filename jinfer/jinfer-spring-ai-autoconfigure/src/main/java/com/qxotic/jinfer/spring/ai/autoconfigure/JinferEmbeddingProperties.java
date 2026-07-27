package com.qxotic.jinfer.spring.ai.autoconfigure;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.context.properties.bind.DefaultValue;

/**
 * Configuration properties for jinfer embeddings, bound under {@code spring.ai.jinfer.embedding}
 * (constructor binding).
 *
 * @param modelPath path to the embedding GGUF (e.g. a Qwen3-Embedding model); required
 * @param contextLength the packing window and per-segment ceiling (default 2048): larger packs more
 *     segments per forward pass and admits longer segments, at the cost of a bigger resident KV
 *     state; {@code <= 0} = the model's own maximum
 */
@ConfigurationProperties("spring.ai.jinfer.embedding")
public record JinferEmbeddingProperties(
        String modelPath, @DefaultValue("2048") int contextLength) {}
