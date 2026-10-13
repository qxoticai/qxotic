package com.qxotic.jinfer.spring.ai.autoconfigure;

import java.util.Map;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.context.properties.bind.ConstructorBinding;
import org.springframework.boot.context.properties.bind.DefaultValue;

/**
 * Configuration properties for jinfer embeddings, bound under {@code spring.ai.jinfer.embedding}
 * (constructor binding).
 *
 * @param model the embedding GGUF (e.g. a Qwen3-Embedding model) as a local path or model ref;
 *     required. A remote ref resolves at context startup
 * @param companions capability to file: media for a multimodal embedder's projector (EmbeddingGemma
 *     2's mmproj); values take the same path-or-ref form as model
 * @param contextCapacity upper bound on the packing window and each embedded sequence (default
 *     2048): larger packs more sequences per forward pass and admits longer sequences, at the cost
 *     of a bigger resident state; 0 uses the model's maxContextLength; negative values are rejected
 */
@ConfigurationProperties("spring.ai.jinfer.embedding")
public record JinferEmbeddingProperties(
        String model, Map<String, String> companions, @DefaultValue("2048") int contextCapacity) {

    @ConstructorBinding
    public JinferEmbeddingProperties {}

    /** The published two-property form: no companions. */
    public JinferEmbeddingProperties(String model, int contextCapacity) {
        this(model, null, contextCapacity);
    }
}
