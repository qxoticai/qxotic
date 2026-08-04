package com.qxotic.jinfer.media;

import com.qxotic.jinfer.Media;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.stream.Stream;

/**
 * Samples a video into a {@link com.qxotic.jinfer.Media.Video} with ffmpeg - the only backend here,
 * because the JDK cannot demux mp4/webm. Frames are taken at a fixed rate, decoded through {@link
 * ImageCodec} (so they inherit its RGB/[0,1]/HWC contract), and capped so the per-frame image
 * tokens cannot blow the context. No audio track.
 */
public final class VideoCodec {

    private VideoCodec() {}

    /**
     * Default frame count, matching the reference processor: N frames sampled UNIFORMLY across the
     * whole duration (never "the first N seconds"), each carrying its true timestamp.
     */
    public static final int DEFAULT_NUM_FRAMES = 32;

    /** The container's duration in seconds (ffprobe - ships with ffmpeg). */
    public static float duration(Path video) throws IOException {
        ProcessBuilder pb =
                new ProcessBuilder(
                        "ffprobe",
                        "-v",
                        "error",
                        "-show_entries",
                        "format=duration",
                        "-of",
                        "csv=p=0",
                        video.toString());
        pb.redirectErrorStream(true);
        try {
            Process p = pb.start();
            String out = new String(p.getInputStream().readAllBytes()).strip();
            if (!p.waitFor(30, java.util.concurrent.TimeUnit.SECONDS) || p.exitValue() != 0) {
                throw new IOException("ffprobe failed for " + video + ": " + out);
            }
            float d = Float.parseFloat(out);
            if (!(d > 0)) throw new IOException("ffprobe reported non-positive duration: " + out);
            return d;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("interrupted probing " + video, e);
        } catch (NumberFormatException e) {
            throw new IOException("unparsable ffprobe duration for " + video, e);
        }
    }

    /**
     * Frames at explicit {@code timestamps} (seconds, ascending) - THE sampling primitive: any
     * policy is a timestamp list (uniform, scene cuts, windowed chunks of a long source,
     * caller-curated key moments). Each timestamp seeks input-side ({@code -ss} before {@code -i}:
     * keyframe-fast, exact enough for frame sampling) and decodes one frame; the result carries the
     * pairs as {@link Media.Video.Frame}.
     */
    public static Media.Video at(Path video, float... timestamps) throws IOException {
        if (timestamps.length == 0) throw new IllegalArgumentException("no timestamps");
        Path dir = Files.createTempDirectory("jinfer-video");
        try {
            Media.Video.Frame[] frames = new Media.Video.Frame[timestamps.length];
            for (int k = 0; k < timestamps.length; k++) {
                Path png = dir.resolve("f" + k + ".png");
                runFfmpeg(
                        "ffmpeg",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-ss",
                        String.format(java.util.Locale.ROOT, "%.3f", timestamps[k]),
                        "-i",
                        video.toString(),
                        "-frames:v",
                        "1",
                        png.toString());
                if (!Files.exists(png)) {
                    throw new IOException(
                            "ffmpeg extracted no frame at " + timestamps[k] + "s from " + video);
                }
                frames[k] = new Media.Video.Frame(ImageCodec.load(png), timestamps[k]);
            }
            return new Media.Video(frames);
        } finally {
            deleteRecursive(dir);
        }
    }

    /** The reference policy: {@link #DEFAULT_NUM_FRAMES} uniform across the whole duration. */
    public static Media.Video sample(Path video) throws IOException {
        return sample(video, DEFAULT_NUM_FRAMES);
    }

    /**
     * {@code numFrames} sampled uniformly across the WHOLE video: {@code t_k = k * duration / n}
     * (the reference sampler's arithmetic) - a one-hour source yields frames ~113s apart with true
     * positions, never "the first n seconds".
     */
    public static Media.Video sample(Path video, int numFrames) throws IOException {
        if (numFrames <= 0) throw new IllegalArgumentException("numFrames must be positive");
        float duration = duration(video);
        float[] timestamps = new float[numFrames];
        for (int k = 0; k < numFrames; k++) timestamps[k] = k * duration / numFrames;
        return at(video, timestamps);
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
