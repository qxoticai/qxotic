package com.qxotic.jinfer.boundary;

import java.util.Optional;

/** Optional capability for models with attached media projectors. */
public interface Multimodal {

    <R extends Media> Optional<MediaProjector<R>> projector(Class<R> modality);
}
