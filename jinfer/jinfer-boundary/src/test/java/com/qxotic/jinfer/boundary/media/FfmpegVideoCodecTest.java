package com.qxotic.jinfer.boundary.media;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.time.Duration;
import org.junit.jupiter.api.Test;

class FfmpegVideoCodecTest {

    @Test
    void seekTimeNeverRoundsPastTheRequestedFrame() {
        assertEquals("1.166", FfmpegVideoCodec.seekTime(Duration.ofNanos(1_166_666_667)));
        assertEquals("0.018", FfmpegVideoCodec.seekTime(Duration.ofNanos(18_750_000)));
    }
}
