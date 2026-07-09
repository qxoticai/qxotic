package com.qxotic.jinfer;

import java.util.Optional;
import java.util.Set;

/**
 * Optional capability: the model accepts non-text input. A model implements this <em>iff</em> it
 * carries media converters — {@code instanceof MultiModal} IS the test, the same convention as
 * {@link MultiToken} (no sentinel). {@link #embedder} hands back a model-wired {@link Embedder} for
 * a modality, which owns its scratch and emits model-dim rows fed as {@link
 * Batch.Input.Embeddings}.
 *
 * <p>Text is intentionally absent: it is the intrinsic modality every model has, consumed as ids
 * via {@link Batch.Input.Tokens} (its converter is the tokenizer). The members here are the
 * continuous modalities, whose converters land at rows instead of ids.
 */
public interface MultiModal {

    /** The media modalities this model can consume — the valid keys for {@link #embedder}. */
    Set<Class<? extends Media>> modalities();

    /**
     * The model-paired converter for {@code modality}, or empty if this model doesn't carry it.
     * Type-safe via the self-typed key: {@code embedder(Media.Audio.class)} returns {@code
     * Embedder<Media.Audio>}.
     */
    <R extends Media> Optional<Embedder<R>> embedder(Class<R> modality);
}
