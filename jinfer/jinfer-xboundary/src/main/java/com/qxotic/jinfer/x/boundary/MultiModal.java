package com.qxotic.jinfer.x.boundary;

import java.util.Optional;
import java.util.Set;

/** Optional capability for x-native models that carry media embedders. */
public interface MultiModal {

    /** The media modalities accepted by this model. */
    Set<Class<? extends Media>> modalities();

    /** Returns the model-paired embedder for {@code modality}, if present. */
    <R extends Media> Optional<Embedder<R>> embedder(Class<R> modality);

    /** Stable identity of every preprocessing choice that can change encoder output. */
    default String encodePlanId() {
        return "";
    }
}
