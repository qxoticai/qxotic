package com.qxotic.jinfer.media;

import com.qxotic.jota.memory.MemoryView;
import java.util.function.Consumer;

/** Converts decoded media into model-specific context rows. */
public interface MediaProjector<R extends Media> {

    /** Number of context rows {@link #project} will produce for {@code media}. */
    int positions(R media);

    void project(R media, int maxChunkSize, Consumer<MemoryView<?>> sink);

    /** Stable identity of preprocessing choices that affect context-cache identity. */
    default String planId() {
        return "";
    }
}
