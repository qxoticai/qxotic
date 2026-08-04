package com.qxotic.jinfer.media;

import com.qxotic.jinfer.Media;
import java.io.IOException;
import java.nio.file.Path;
import java.time.Duration;

/**
 * Samples a video into a {@link com.qxotic.jinfer.Media.Video}. A backend supplies three primitives
 * - {@link #totalDuration}, {@link #framePeriod}, and {@link #framesAt} (frames at explicit
 * timestamps; {@link #frameAt} is its correctly-shaped singular) - and the sampling policies are
 * default methods of pure timestamp arithmetic on top: {@link #uniform} takes n equal-segment
 * representatives (centered), {@link #span} takes n frames end-to-end inclusive (fencepost) - both
 * over the whole source or a {@code [from, to]} window. The one shipped backend is {@link
 * #ffmpeg()} (the JDK cannot demux mp4/webm). Frames decode through {@link ImageCodec} (inheriting
 * its RGB/[0,1]/HWC contract) and carry their true timestamps. No audio track.
 *
 * <p>DELIBERATE SHAPE - this is a SAMPLER, not a decoder, and its asymmetry with {@link
 * ImageDecoder}/{@link AudioDecoder} (whole payload in, media out) is intentional: "decode the
 * whole video" is not a meaningful operation at these scales, and any default rate would be a
 * policy smuggled into a signature (the original fps-based loader silently took the FIRST n seconds
 * - the exact trap). Equally deliberate exclusions: no {@code byte[]} overloads until a caller
 * actually holds bytes (ffmpeg needs a file; the wire that receives bytes owns that temp file
 * today), no {@code first}/{@code last} conveniences ({@link #framesAt} composes every policy a
 * caller can state as timestamps - and "the first n frames" at native rate is the first-N-seconds
 * trap in method form), and nothing model-shaped - token budgets, frame markers, timestamp
 * formatting are template concerns. The contract ends at truthfully timestamped frames.
 */
public interface VideoCodec {

    /**
     * Default frame count, matching the reference processor: N frames sampled UNIFORMLY across the
     * whole duration (never "the first N seconds"), each carrying its true timestamp.
     */
    int DEFAULT_NUM_FRAMES = 32;

    /** The shipped backend: shells out to ffmpeg/ffprobe on PATH. */
    static VideoCodec ffmpeg() {
        return FfmpegVideoCodec.INSTANCE;
    }

    /** Short backend name, for logging/diagnostics ("ffmpeg"). */
    String name();

    /** The container's total duration. */
    Duration totalDuration(Path video) throws IOException;

    /**
     * The native stream's frame period (1/fps) - the gap between the source's last frame and its
     * duration, which is why {@link #span} needs it: seeking past the final frame yields nothing.
     */
    Duration framePeriod(Path video) throws IOException;

    /**
     * Frames at explicit {@code timestamps} (ascending) - THE sampling primitive: any policy is a
     * timestamp list (uniform, scene cuts, windowed chunks of a long source, caller-curated key
     * moments). Each timestamp decodes exactly one frame; the result carries the pairs as {@link
     * Media.Video.Frame}. The singular counterpart is {@link #frameAt}.
     */
    Media.Video framesAt(Path video, Duration... timestamps) throws IOException;

    /**
     * The single frame at {@code timestamp} - the singular of {@link #framesAt}, returning the
     * correctly shaped {@link Media.Video.Frame} (one frame is not a video).
     */
    default Media.Video.Frame frameAt(Path video, Duration timestamp) throws IOException {
        return framesAt(video, timestamp).frames().get(0);
    }

    /** {@link #DEFAULT_NUM_FRAMES} equal-segment representatives across the duration. */
    default Media.Video uniform(Path video) throws IOException {
        return uniform(video, DEFAULT_NUM_FRAMES);
    }

    /**
     * {@code n} equal-segment representatives across the whole source: {@link #uniform(Path,
     * Duration, Duration, int)} over {@code [0, totalDuration]}. {@code n=1} is the middle frame
     * (the most representative single sample), {@code n=3} is quarter points 1/6, 1/2, 5/6.
     */
    default Media.Video uniform(Path video, int n) throws IOException {
        return uniform(video, Duration.ZERO, totalDuration(video), n);
    }

    /**
     * {@code n} frames, each representing an equal 1/n segment of the window {@code [from, to]}:
     * the CENTERED scheme {@code t_k = from + (k + 1/2) * (to - from) / n}. The ends are covered to
     * within {@code (to - from) / 2n} - use {@link #span} when the literal boundary frames matter.
     * {@code to} is clamped to the source's duration (unlike {@link #span}'s last-frame clamp: the
     * centers are interior, so the duration itself is a valid right edge). (Deliberate deviation
     * from the HF reference's start-aligned {@code arange(0, total, total/num)}: that scheme never
     * sees the final {@code duration/n} of the source and its n=1 is the first frame; the
     * interleaved true timestamps ground either scheme for the model.)
     */
    default Media.Video uniform(Path video, Duration from, Duration to, int n) throws IOException {
        if (n <= 0) throw new IllegalArgumentException("n must be positive");
        long start = from.toNanos();
        long end = Math.min(to.toNanos(), totalDuration(video).toNanos());
        if (start < 0 || start >= end) {
            throw new IllegalArgumentException(
                    "uniform window must satisfy 0 <= from < to (clamped): from="
                            + from
                            + " to="
                            + to);
        }
        long window = end - start;
        Duration[] timestamps = new Duration[n];
        for (int k = 0; k < n; k++)
            timestamps[k] = Duration.ofNanos(start + k * (window / n) + window / (2L * n));
        return framesAt(video, timestamps);
    }

    /**
     * {@code n} frames from start to end INCLUSIVE: {@link #span(Path, Duration, Duration, int)}
     * over the whole source. {@code n=2} is the first and last frames, {@code n=3} adds the middle.
     */
    default Media.Video span(Path video, int n) throws IOException {
        return span(video, Duration.ZERO, totalDuration(video), n);
    }

    /**
     * {@code n} frames across the window {@code [from, to]} INCLUSIVE: the fencepost scheme {@code
     * t_k = from + k * (to - from) / (n-1)}. {@code to} is clamped to the source's final frame
     * ({@code duration - framePeriod} - seeking past it yields nothing; the end of a video is a
     * frame, not an instant), so {@code to = totalDuration} means "through the last frame". Needs
     * {@code n >= 2}: "both ends" is the contract - one frame has no ends (use {@link #frameAt} or
     * {@link #uniform}).
     */
    default Media.Video span(Path video, Duration from, Duration to, int n) throws IOException {
        if (n < 2) throw new IllegalArgumentException("span needs n >= 2 (both ends inclusive)");
        long start = from.toNanos();
        long end =
                Math.min(
                        to.toNanos(),
                        totalDuration(video).toNanos() - framePeriod(video).toNanos());
        if (start < 0 || start >= end) {
            throw new IllegalArgumentException(
                    "span window must satisfy 0 <= from < to (clamped): from="
                            + from
                            + " to="
                            + to);
        }
        Duration[] timestamps = new Duration[n];
        for (int k = 0; k < n; k++)
            timestamps[k] = Duration.ofNanos(start + k * ((end - start) / (n - 1)));
        return framesAt(video, timestamps);
    }
}
