package com.qxotic.jinfer.media;

import java.util.Optional;

/**
 * Optional capability for models with attached media projectors. Implementing it says the family
 * can take media, not that this instance can: a checkpoint loaded without its projector file still
 * implements it, with no projector. Ask {@link #supports} which modalities this instance reads.
 */
public interface Multimodal {

    /** The projector for one modality, or empty when this model cannot ingest it. */
    <R extends Media> Optional<MediaProjector<R>> projector(Class<R> modality);

    /**
     * Whether this instance reads {@code modality} at all, projector or not: a family that reads a
     * video by sampling its frames through the vision tower has no {@link Media.Video} projector,
     * yet it takes video. Every modality a family reads through a projector keeps the default.
     */
    default boolean supports(Class<? extends Media> modality) {
        return projector(modality).isPresent();
    }
}
