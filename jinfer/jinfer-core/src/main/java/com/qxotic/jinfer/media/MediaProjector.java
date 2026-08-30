package com.qxotic.jinfer.media;

import com.qxotic.jinfer.Batch;
import com.qxotic.jota.memory.MemoryView;
import java.util.function.Consumer;

/** Converts decoded media into model-specific context rows. */
public interface MediaProjector<R extends Media> {

    /** Number of context rows {@link #project} will produce for {@code media}. */
    int positions(R media);

    void project(R media, int maxChunkSize, Consumer<MemoryView<?>> sink);

    /** Optional decoder positions for all rows, in the same order as {@link #project}. */
    default Batch.Positions decoderPositions(R media) {
        return null;
    }

    /** Stable identity of preprocessing choices that affect context-cache identity. */
    default String planId() {
        return "";
    }
}
