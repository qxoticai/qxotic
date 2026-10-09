package com.qxotic.jinfer.codecs;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class FfmpegVideoCodecTest {

    @Test
    void seekTimeNeverRoundsPastTheRequestedFrame() {
        assertEquals("1.166", FfmpegVideoCodec.seekTime(Duration.ofNanos(1_166_666_667)));
        assertEquals("0.018", FfmpegVideoCodec.seekTime(Duration.ofNanos(18_750_000)));
    }

    @Test
    void oneProbeAnswersBothPropertiesUntilTheFileChanges(@TempDir Path dir) throws Exception {
        Assumptions.assumeTrue(onPath("ffmpeg") && onPath("ffprobe"), "needs ffmpeg and ffprobe");
        Path video = dir.resolve("clip.mkv");
        FfmpegVideoCodec codec = new FfmpegVideoCodec();
        clip(video, 2, 10);
        assertEquals(Duration.ofSeconds(2), codec.totalDuration(video));
        assertEquals(Duration.ofMillis(100), codec.framePeriod(video));
        clip(video, 3, 5); // the same path rewritten: probed afresh
        assertEquals(Duration.ofSeconds(3), codec.totalDuration(video));
        assertEquals(Duration.ofMillis(200), codec.framePeriod(video));
    }

    private static void clip(Path video, int seconds, int fps) throws Exception {
        Process p =
                new ProcessBuilder(
                                List.of(
                                        "ffmpeg",
                                        "-y",
                                        "-loglevel",
                                        "error",
                                        "-f",
                                        "lavfi",
                                        "-i",
                                        "testsrc=size=64x48:rate=" + fps + ":duration=" + seconds,
                                        "-c:v",
                                        "ffv1",
                                        video.toString()))
                        .redirectErrorStream(true)
                        .start();
        p.getInputStream().readAllBytes();
        assertEquals(0, p.waitFor(), "ffmpeg");
    }

    private static boolean onPath(String tool) {
        try {
            Process p = new ProcessBuilder(tool, "-version").redirectErrorStream(true).start();
            p.getInputStream().readAllBytes();
            return p.waitFor() == 0;
        } catch (Exception absent) {
            return false;
        }
    }
}
