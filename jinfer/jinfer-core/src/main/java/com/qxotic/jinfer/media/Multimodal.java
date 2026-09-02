package com.qxotic.jinfer.media;

import java.util.Optional;

/** Optional capability for models with attached media projectors. */
public interface Multimodal {

    /** The projector for one modality, or empty when this model cannot ingest it. */
    <R extends Media> Optional<MediaProjector<R>> projector(Class<R> modality);
}
