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
 * The {@link VideoCodec} primitives via ffmpeg/ffprobe on PATH: {@code totalDuration} and {@code
 * framePeriod} probe the container ({@code -of csv=p=0}), and each {@code framesAt} timestamp seeks
 * input-side ({@code -ss} before {@code -i}: keyframe-fast, exact enough for frame sampling) and
 * decodes one frame to a temp PNG read back through {@link ImageCodec}. Stateless; loud on failure.
 */
public final class FfmpegVideoCodec implements VideoCodec {

    static final FfmpegVideoCodec INSTANCE = new FfmpegVideoCodec();

    @Override
    public String name() {
        return "ffmpeg";
    }

    @Override
    public Duration totalDuration(Path video) throws IOException {
        String out = probe(video, "-show_entries", "format=duration");
        try {
            double d = Double.parseDouble(out);
            if (!(d > 0)) throw new IOException("ffprobe reported non-positive duration: " + out);
            return Duration.ofNanos((long) (d * 1e9));
        } catch (NumberFormatException e) {
            throw new IOException("unparsable ffprobe duration for " + video + ": " + out);
        }
    }

    @Override
    public Duration framePeriod(Path video) throws IOException {
        String out = probe(video, "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate");
        try {
            int slash = out.indexOf('/');
            float fps =
                    slash < 0
                            ? Float.parseFloat(out)
                            : Float.parseFloat(out.substring(0, slash))
                                    / Float.parseFloat(out.substring(slash + 1));
            if (!(fps > 0)) throw new IOException("non-positive frame rate: " + out);
            return Duration.ofNanos((long) (1e9 / fps));
        } catch (NumberFormatException e) {
            throw new IOException("unparsable ffprobe frame rate for " + video + ": " + out);
        }
    }

    @Override
    public Media.Video framesAt(Path video, Duration... timestamps) throws IOException {
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
