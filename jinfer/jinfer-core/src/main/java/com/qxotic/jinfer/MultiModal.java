package com.qxotic.jinfer;

import java.util.Optional;
import java.util.Set;

/**
 * Optional capability: the model accepts non-text input. A model implements this <em>iff</em> it
 * carries media converters — {@code instanceof MultiModal} IS the test (no sentinel). {@link
 * #embedder} hands back a model-wired {@link Embedder} for a modality, which owns its scratch and
 * emits model-dim rows fed as {@link Batch.Input.Embeddings}.
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

    /**
     * A canonical string naming every parameter that changes the encoder's OUTPUT for the same
     * input - preprocessing knobs (token budgets, resize modes) plus an explicit version token
     * bumped whenever the plan's code changes. Folded into the model's cache seed when a media
     * projector loads, so content-keyed media blocks (keyed by SOURCE bytes, not encoded rows - see
     * {@code Batch.embeddings}) can never match across a plan change the bytes cannot see.
     */
    default String encodePlanId() {
        return "";
    }
}
