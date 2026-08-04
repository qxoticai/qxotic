package com.qxotic.jinfer.media;

import com.qxotic.jinfer.Media;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Stream;

/**
 * Samples a video into a {@link com.qxotic.jinfer.Media.Video} with ffmpeg - the only backend here,
 * because the JDK cannot demux mp4/webm. One primitive and three named policies: {@link #framesAt}
 * takes frames at explicit timestamps (any sampling policy is a timestamp list; {@link #frameAt} is
 * its correctly-shaped singular), {@link #uniform} takes n equal-segment representatives
 * (centered), {@link #span} takes n frames start-to-end inclusive (fencepost), {@link #first} takes
 * the opening n frames at native rate. Frames decode through {@link ImageCodec} (inheriting its
 * RGB/[0,1]/HWC contract) and carry their true timestamps. No audio track.
 *
 * <p>DELIBERATE SHAPE - this class is a SAMPLER, not a decoder, and its asymmetry with {@link
 * ImageCodec}/{@link AudioCodec} (whole payload in, media out) is intentional: "decode the whole
 * video" is not a meaningful operation at these scales, and any default rate would be a policy
 * smuggled into a signature (the original fps-based loader silently took the FIRST n seconds - the
 * exact trap). Equally deliberate exclusions: no {@code byte[]} overloads until a caller actually
 * holds bytes (ffmpeg needs a file; the wire that receives bytes owns that temp file today), no
 * {@code window}/{@code last} conveniences ({@link #framesAt} composes every policy a caller can
 * state as timestamps), and nothing model-shaped - token budgets, frame markers, timestamp
 * formatting are template concerns. The contract ends at truthfully timestamped frames.
 */
public final class VideoCodec {

    private VideoCodec() {}

    /**
     * Default frame count, matching the reference processor: N frames sampled UNIFORMLY across the
     * whole duration (never "the first N seconds"), each carrying its true timestamp.
     */
    public static final int DEFAULT_NUM_FRAMES = 32;

    /** The container's total duration (ffprobe - ships with ffmpeg). */
    public static Duration totalDuration(Path video) throws IOException {
        String out = probe(video, "-show_entries", "format=duration");
        try {
            double d = Double.parseDouble(out);
            if (!(d > 0)) throw new IOException("ffprobe reported non-positive duration: " + out);
            return Duration.ofNanos((long) (d * 1e9));
        } catch (NumberFormatException e) {
            throw new IOException("unparsable ffprobe duration for " + video + ": " + out);
        }
    }

    /** One ffprobe field as text ({@code -of csv=p=0}); loud on failure. */
    private static String probe(Path video, String... entries) throws IOException {
        List<String> cmd = new ArrayList<>(List.of("ffprobe", "-v", "error"));
        cmd.addAll(List.of(entries));
        cmd.addAll(List.of("-of", "csv=p=0", video.toString()));
        ProcessBuilder pb = new ProcessBuilder(cmd);
        pb.redirectErrorStream(true);
        try {
            Process p = pb.start();
            String out = new String(p.getInputStream().readAllBytes()).strip();
            if (!p.waitFor(30, java.util.concurrent.TimeUnit.SECONDS) || p.exitValue() != 0) {
                throw new IOException("ffprobe failed for " + video + ": " + out);
            }
            return out;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("interrupted probing " + video, e);
        }
    }

    /**
     * Frames at explicit {@code timestamps} (ascending) - THE sampling primitive: any policy is a
     * timestamp list (uniform, scene cuts, windowed chunks of a long source, caller-curated key
     * moments). Each timestamp seeks input-side ({@code -ss} before {@code -i}: keyframe-fast,
     * exact enough for frame sampling) and decodes one frame; the result carries the pairs as
     * {@link Media.Video.Frame}. The singular counterpart is {@link #frameAt}.
     */
    public static Media.Video framesAt(Path video, Duration... timestamps) throws IOException {
        if (timestamps.length == 0) throw new IllegalArgumentException("no timestamps");
        Path dir = Files.createTempDirectory("jinfer-video");
        try {
            List<Media.Video.Frame> frames = new ArrayList<>(timestamps.length);
            for (int k = 0; k < timestamps.length; k++) {
                Path png = dir.resolve("f" + k + ".png");
                double seconds = timestamps[k].toNanos() / 1e9;
                runFfmpeg(
                        "ffmpeg",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-ss",
                        String.format(java.util.Locale.ROOT, "%.3f", seconds),
                        "-i",
                        video.toString(),
                        "-frames:v",
                        "1",
                        png.toString());
                if (!Files.exists(png)) {
                    throw new IOException(
                            "ffmpeg extracted no frame at " + timestamps[k] + " from " + video);
                }
                frames.add(new Media.Video.Frame(ImageCodec.load(png), timestamps[k]));
            }
            return new Media.Video(frames);
        } finally {
            deleteRecursive(dir);
        }
    }

    /**
     * The single frame at {@code timestamp} - the singular of {@link #framesAt}, returning the
     * correctly shaped {@link Media.Video.Frame} (one frame is not a video).
     */
    public static Media.Video.Frame frameAt(Path video, Duration timestamp) throws IOException {
        return framesAt(video, timestamp).frames().get(0);
    }

    /** {@link #DEFAULT_NUM_FRAMES} equal-segment representatives across the duration. */
    public static Media.Video uniform(Path video) throws IOException {
        return uniform(video, DEFAULT_NUM_FRAMES);
    }

    /**
     * {@code n} frames, each representing an equal 1/n segment of the source: the CENTERED scheme
     * {@code t_k = (k + 1/2) * duration / n}. {@code n=1} is the middle frame (the most
     * representative single sample - "the first frame" is {@link #first}'s job), {@code n=3} is
     * quarter points 1/6, 1/2, 5/6; the ends are covered to within {@code duration/2n}. Use {@link
     * #span} when the literal first and last frames matter. (Deliberate deviation from the HF
     * reference's start-aligned {@code arange(0, total, total/num)}: that scheme never sees the
     * final {@code duration/n} of the source and its n=1 duplicates {@code first(1)}; the
     * interleaved true timestamps ground either scheme for the model.)
     */
    public static Media.Video uniform(Path video, int n) throws IOException {
        if (n <= 0) throw new IllegalArgumentException("n must be positive");
        long nanos = totalDuration(video).toNanos();
        Duration[] timestamps = new Duration[n];
        for (int k = 0; k < n; k++)
            timestamps[k] = Duration.ofNanos(k * (nanos / n) + nanos / (2L * n));
        return framesAt(video, timestamps);
    }

    /**
     * {@code n} frames from start to end INCLUSIVE: the fencepost scheme {@code t_k = k * last /
     * (n-1)} where {@code last = duration - 1/fps} is the true timestamp of the source's final
     * frame (seeking past it yields nothing - the end of a video is a frame, not an instant).
     * {@code n=2} is the first and last frames, {@code n=3} adds the middle. Needs {@code n >= 2}:
     * "both ends" is the contract - one frame has no ends (use {@link #frameAt} or {@link
     * #uniform}).
     */
    public static Media.Video span(Path video, int n) throws IOException {
        if (n < 2) throw new IllegalArgumentException("span needs n >= 2 (both ends inclusive)");
        long frameNanos = (long) (1e9 / nativeFps(video));
        long last = totalDuration(video).toNanos() - frameNanos;
        Duration[] timestamps = new Duration[n];
        for (int k = 0; k < n; k++) timestamps[k] = Duration.ofNanos(k * (last / (n - 1)));
        return framesAt(video, timestamps);
    }

    /**
     * The source's FIRST {@code n} frames in decode order, stamped at their native times ({@code
     * t_k = k / fps}). Dense temporal detail at the start, no coverage beyond it - the dual of
     * {@link #uniform}; the opening-moments policy (previews, title cards, "how does this begin").
     * A source with fewer frames yields what it has. Cheapest extraction: one pass, no seeks.
     */
    public static Media.Video first(Path video, int n) throws IOException {
        if (n <= 0) throw new IllegalArgumentException("n must be positive");
        float fps = nativeFps(video);
        Path dir = Files.createTempDirectory("jinfer-video");
        try {
            runFfmpeg(
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    video.toString(),
                    "-frames:v",
                    String.valueOf(n),
                    dir.resolve("f%06d.png").toString());
            List<Path> pngs;
            try (Stream<Path> s = Files.list(dir)) {
                pngs =
                        s.filter(f -> f.getFileName().toString().endsWith(".png"))
                                .sorted(Comparator.comparing(Path::getFileName))
                                .toList();
            }
            if (pngs.isEmpty()) throw new IOException("ffmpeg extracted no frames from " + video);
            List<Media.Video.Frame> frames = new ArrayList<>(pngs.size());
            for (int k = 0; k < pngs.size(); k++) {
                Duration t = Duration.ofNanos((long) (k / fps * 1e9));
                frames.add(new Media.Video.Frame(ImageCodec.load(pngs.get(k)), t));
            }
            return new Media.Video(frames);
        } finally {
            deleteRecursive(dir);
        }
    }

    /** The video stream's native frame rate (ffprobe r_frame_rate, a fraction like 30000/1001). */
    private static float nativeFps(Path video) throws IOException {
        String out = probe(video, "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate");
        try {
            int slash = out.indexOf('/');
            float fps =
                    slash < 0
                            ? Float.parseFloat(out)
                            : Float.parseFloat(out.substring(0, slash))
                                    / Float.parseFloat(out.substring(slash + 1));
            if (!(fps > 0)) throw new IOException("non-positive frame rate: " + out);
            return fps;
        } catch (NumberFormatException e) {
            throw new IOException("unparsable ffprobe frame rate for " + video + ": " + out);
        }
    }

    private static void runFfmpeg(String... cmd) throws IOException {
        Process p;
        try {
            p = new ProcessBuilder(cmd).redirectError(ProcessBuilder.Redirect.INHERIT).start();
            p.getOutputStream().close();
            p.getInputStream().readAllBytes();
        } catch (IOException e) {
            throw new IOException("failed to launch ffmpeg (is it on PATH?): " + e.getMessage(), e);
        }
        int code;
        try {
            code = p.waitFor();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("interrupted waiting for ffmpeg", e);
        }
        if (code != 0) throw new IOException("ffmpeg exited " + code);
    }

    private static void deleteRecursive(Path dir) throws IOException {
        try (Stream<Path> s = Files.walk(dir)) {
            s.sorted(Comparator.reverseOrder())
                    .forEach(
                            p -> {
                                try {
                                    Files.deleteIfExists(p);
                                } catch (IOException ignored) {
                                }
                            });
        }
    }
}
