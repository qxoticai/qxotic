package com.qxotic.jinfer.media;

import com.qxotic.jinfer.Media;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.List;
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

    /** Sample {@link #DEFAULT_NUM_FRAMES} frames uniformly across the video. */
    public static Media.Video load(Path path) throws IOException {
        return load(path, DEFAULT_NUM_FRAMES);
    }

    /**
     * Sample {@code numFrames} frames uniformly across the WHOLE video into a {@link Media.Video}
     * (audio dropped). Mirrors the reference sampler ({@code indices = arange(0, total,
     * total/num)}): a one-hour video yields frames ~113s apart, timestamped at their true positions
     * - the interleaved MM:SS stamps are what keep sparse sampling temporally grounded. A video
     * shorter than {@code numFrames} frames yields what it has.
     */
    public static Media.Video load(Path path, int numFrames) throws IOException {
        if (numFrames <= 0) throw new IllegalArgumentException("numFrames must be positive");
        double duration = probeDurationSeconds(path);
        Path dir = Files.createTempDirectory("jinfer-video");
        try {
            // one pass at rate numFrames/duration = evenly spaced samples starting at ~t=0
            runFfmpeg(
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    path.toString(),
                    "-vf",
                    String.format(java.util.Locale.ROOT, "fps=%.8f", numFrames / duration),
                    "-frames:v",
                    String.valueOf(numFrames),
                    dir.resolve("f%04d.png").toString());
            List<Path> pngs;
            try (Stream<Path> s = Files.list(dir)) {
                pngs =
                        s.filter(p -> p.getFileName().toString().endsWith(".png"))
                                .sorted(Comparator.comparing(Path::getFileName))
                                .toList();
            }
            if (pngs.isEmpty()) throw new IOException("ffmpeg extracted no frames from " + path);
            // true positions of the uniform samples: t_k = k * duration / n (reference-aligned)
            int n = pngs.size();
            Media.Video.Frame[] frames = new Media.Video.Frame[n];
            for (int k = 0; k < n; k++) {
                frames[k] =
                        new Media.Video.Frame(
                                ImageCodec.load(pngs.get(k)), (float) (k * duration / n));
            }
            return new Media.Video(frames);
        } finally {
            deleteRecursive(dir);
        }
    }

    /** The container's duration via ffprobe (ships with ffmpeg) - needed to spread the samples. */
    private static double probeDurationSeconds(Path path) throws IOException {
        ProcessBuilder pb =
                new ProcessBuilder(
                        "ffprobe",
                        "-v",
                        "error",
                        "-show_entries",
                        "format=duration",
                        "-of",
                        "csv=p=0",
                        path.toString());
        pb.redirectErrorStream(true);
        try {
            Process p = pb.start();
            String out = new String(p.getInputStream().readAllBytes()).strip();
            if (!p.waitFor(30, java.util.concurrent.TimeUnit.SECONDS) || p.exitValue() != 0) {
                throw new IOException("ffprobe failed for " + path + ": " + out);
            }
            double d = Double.parseDouble(out);
            if (!(d > 0)) throw new IOException("ffprobe reported non-positive duration: " + out);
            return d;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("interrupted probing " + path, e);
        } catch (NumberFormatException e) {
            throw new IOException("unparsable ffprobe duration for " + path, e);
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
