package com.qxotic.jinfer;

import com.qxotic.jota.memory.MemoryView;
import java.util.function.Consumer;

/** Converts decoded media into model-specific context rows. */
public interface MediaProjector<R extends Media> {

    void project(R media, int maxChunkSize, Consumer<MemoryView<?>> sink);

    default int positions(R media) {
        throw new UnsupportedOperationException("projected position count is unavailable");
    }

    /** Stable identity of preprocessing choices that affect context-cache identity. */
    default String planId() {
        return "";
    }
}
