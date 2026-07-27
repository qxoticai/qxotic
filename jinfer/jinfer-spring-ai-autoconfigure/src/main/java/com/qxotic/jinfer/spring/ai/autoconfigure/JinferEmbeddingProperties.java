package com.qxotic.jinfer.spring.ai.autoconfigure;

import org.springframework.boot.context.properties.ConfigurationProperties;

/**
 * Configuration properties for jinfer embeddings, bound under {@code spring.ai.jinfer.embedding}.
 */
@ConfigurationProperties("spring.ai.jinfer.embedding")
public class JinferEmbeddingProperties {

    /** Path to the embedding GGUF (e.g. a Qwen3-Embedding model). Required. */
    private String modelPath;

    /**
     * The packing window and per-segment ceiling (default 2048): larger packs more segments per
     * forward pass and admits longer segments, at the cost of a bigger resident KV state. {@code <=
     * 0} = the model's own maximum.
     */
    private int contextLength = 2048;

    public String getModelPath() {
        return modelPath;
    }

    public void setModelPath(String modelPath) {
        this.modelPath = modelPath;
    }

    public int getContextLength() {
        return contextLength;
    }

    public void setContextLength(int contextLength) {
        this.contextLength = contextLength;
    }
}
