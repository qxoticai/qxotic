package com.qxotic.jinfer.media;

import com.qxotic.jinfer.Media;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Stream;

/**
 * Samples a video into a {@link com.qxotic.jinfer.Media.Video} with ffmpeg - the only backend here,
 * because the JDK cannot demux mp4/webm. One primitive and two named policies: {@link #at} takes
 * frames at explicit timestamps (any sampling policy is a timestamp list), {@link #uniform} spreads
 * n frames across the whole duration (the reference processors' policy), {@link #first} takes the
 * opening n frames at native rate. Frames decode through {@link ImageCodec} (inheriting its
 * RGB/[0,1]/HWC contract) and carry their true timestamps. No audio track.
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
        String out = probe(video, "-show_entries", "format=duration");
        try {
            float d = Float.parseFloat(out);
            if (!(d > 0)) throw new IOException("ffprobe reported non-positive duration: " + out);
            return d;
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

    /** The reference policy: {@link #DEFAULT_NUM_FRAMES} frames uniform across the duration. */
    public static Media.Video uniform(Path video) throws IOException {
        return uniform(video, DEFAULT_NUM_FRAMES);
    }

    /**
     * {@code n} frames at EQUALLY SPACED timestamps across the WHOLE source: {@code t_k = k *
     * duration / n} - the reference processors' arithmetic. Whole-source coverage at sparse
     * temporal detail (an hour = n glimpses ~{@code 3600/n}s apart, true positions, never "the
     * first n seconds"); the dual of {@link #first}.
     */
    public static Media.Video uniform(Path video, int n) throws IOException {
        if (n <= 0) throw new IllegalArgumentException("n must be positive");
        float duration = duration(video);
        float[] timestamps = new float[n];
        for (int k = 0; k < n; k++) timestamps[k] = k * duration / n;
        return at(video, timestamps);
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
            Media.Video.Frame[] frames = new Media.Video.Frame[pngs.size()];
            for (int k = 0; k < pngs.size(); k++) {
                frames[k] = new Media.Video.Frame(ImageCodec.load(pngs.get(k)), k / fps);
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
