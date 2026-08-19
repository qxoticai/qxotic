package com.qxotic.jinfer;

import java.time.Duration;
import java.util.Objects;

/**
 * One frame's cache identity: video key + true timestamp, as a structure - no derived hash, no
 * delimiter grammar. Timestamp, not index: the same instant keys the same pixels under any sampling
 * policy.
 */
public record FrameRef(ContentKey video, Duration timestamp) {

    public FrameRef {
        Objects.requireNonNull(video, "video");
        Objects.requireNonNull(timestamp, "timestamp");
    }

    @Override
    public String toString() {
        return video + "#t=" + timestamp.toNanos();
    }
}
