package com.qxotic.jinfer.codecs;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jinfer.media.Media;
import java.nio.file.Path;
import java.time.Duration;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

class VideoCodecSamplingTest {

    /** A source of {@code seconds} at {@code fps} that decodes a blank frame at any timestamp. */
    private record Source(double seconds, int fps) implements VideoCodec {
        @Override
        public String name() {
            return "fake";
        }

        @Override
        public Duration totalDuration(Path video) {
            return Duration.ofNanos(Math.round(seconds * 1e9));
        }

        @Override
        public Duration framePeriod(Path video) {
            return Duration.ofNanos(1_000_000_000L / fps);
        }

        @Override
        public Media.Video framesAt(Path video, Duration... timestamps) {
            Media.Image blank = new Media.Image(new float[3], 1, 1, 3);
            return new Media.Video(
                    Arrays.stream(timestamps).map(t -> new Media.Video.Frame(blank, t)).toList());
        }
    }

    private static List<Double> everySecond(double seconds, int fps) throws Exception {
        return everySecond(seconds, fps, 32);
    }

    private static List<Double> everySecond(double seconds, int fps, int maxFrames)
            throws Exception {
        return new Source(seconds, fps)
                .every(Path.of("video"), Duration.ofSeconds(1), maxFrames).frames().stream()
                        .map(f -> f.timestamp().toNanos() / 1e9)
                        .toList();
    }

    @Test
    void samplesLikeTheReferenceVideoProcessor() throws Exception {
        // EmbeddingGemma2VideoProcessor.sample_frames at fps=1, max_frames=32, uniform overflow
        assertEquals(List.of(0.0, 1.0, 2.0), everySecond(3.5, 30));
        assertEquals(List.of(0.0), everySecond(0.4, 30));
        assertEquals(List.of(0.0), everySecond(40, 24, 1), "linspace's one point is the first");
        assertEquals(List.of(0.0, 39.0), everySecond(40, 24, 2));
        assertEquals(List.of(0.0, 3.0, 6.0, 9.0), everySecond(10, 24, 4));
        assertEquals(32, everySecond(32, 25).size());
        assertEquals(31.0, everySecond(32, 25).getLast());
        assertEquals(
                List.of(
                        0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0, 33.0, 36.0,
                        39.0, 42.0, 45.0, 48.0, 51.0, 54.0, 57.0, 60.0, 63.0, 66.0, 69.0, 72.0,
                        75.0, 78.0, 81.0, 84.0, 87.0, 90.0, 94.0),
                everySecond(95.4, 30));
        assertEquals(
                List.of(
                        0.0, 1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 8.0, 10.0, 11.0, 12.0, 13.0, 15.0, 16.0,
                        17.0, 18.0, 20.0, 21.0, 22.0, 23.0, 25.0, 26.0, 27.0, 28.0, 30.0, 31.0,
                        32.0, 33.0, 35.0, 36.0, 37.0, 39.0),
                everySecond(40, 24));
    }
}
