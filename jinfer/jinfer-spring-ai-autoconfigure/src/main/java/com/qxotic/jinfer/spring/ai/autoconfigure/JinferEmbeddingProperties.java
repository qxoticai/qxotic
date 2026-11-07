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
 * @param contextCapacity upper bound on the packing window and each embedded sequence: 0, the
 *     default, is PromptCache.Options.DEFAULT_CONTEXT_CAPACITY (4096) or the model's
 *     maxContextLength when smaller; -1 is that length; another value is exact, refused at boot
 *     past that length. Larger packs more sequences per forward pass and admits longer sequences,
 *     at the cost of a bigger resident state
 * @param dimensions the embedding width for every call that names none, and the width the model
 *     reports to vector stores: 0, the default, is the model's own; a Matryoshka model takes any
 *     width in its range (EmbeddingGemma 2: 128 to 768, Qwen3-Embedding: 32 to its own), a
 *     fixed-width model its own only, refused at boot otherwise
 */
@ConfigurationProperties("spring.ai.jinfer.embedding")
public record JinferEmbeddingProperties(
        String model,
        Map<String, String> companions,
        @DefaultValue("0") int contextCapacity,
        @DefaultValue("0") int dimensions) {

    @ConstructorBinding
    public JinferEmbeddingProperties {}

    /** The published two-property form: no companions, the model's own width. */
    public JinferEmbeddingProperties(String model, int contextCapacity) {
        this(model, null, contextCapacity, 0);
    }
}
