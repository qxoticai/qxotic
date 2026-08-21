package com.qxotic.jinfer.spring.ai.autoconfigure;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.context.properties.bind.DefaultValue;

/**
 * Configuration properties for jinfer embeddings, bound under {@code spring.ai.jinfer.embedding}
 * (constructor binding).
 *
 * @param model the embedding GGUF (e.g. a Qwen3-Embedding model) as a local path or model ref;
 *     required. A remote ref resolves at context startup
 * @param contextLength upper bound on the packing window and each embedded sequence (default 2048):
 *     larger packs more sequences per forward pass and admits longer sequences, at the cost of a
 *     bigger resident state; {@code 0} uses the model's declared context length; negative values
 *     are rejected
 */
@ConfigurationProperties("spring.ai.jinfer.embedding")
public record JinferEmbeddingProperties(String model, @DefaultValue("2048") int contextLength) {}
