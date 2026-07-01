package com.qxotic.jinfer;

/**
 * An encoder: a {@link Model} backbone whose head produces a pooled representation of the ingested
 * sequence (a sentence/document embedding), not vocabulary logits. Distinct from {@link Embedder},
 * which projects a non-text modality into model-dim input rows — this consumes tokens and outputs an
 * embedding. (Sketch: no GGUF encoder is wired yet.)
 */
public interface EmbeddingModel<C extends Config, W, S extends RuntimeState> extends Model<C, W, S> {
    FloatTensor embedding(S state);
}
