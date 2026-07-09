// Decodes a video file into a Media.Video by sampling frames with ffmpeg. Unlike images/audio there
// is
// no non-ffmpeg fallback (the JDK cannot demux mp4/webm), so this is ffmpeg-only -
// native-image-safe since
// ffmpeg is referenced directly (no reflection, no java.desktop). Frames are extracted at a fixed
// rate,
// each decoded through ImageCodec (so it inherits the same RGB/[0,1]/HWC contract), and capped so
// the
// per-frame image tokens do not blow the context. Audio is not captured (frames only) - a
// follow-up.
package com.qxotic.jinfer.media;

import com.qxotic.jinfer.Media;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Stream;

public final class VideoCodec {

    private VideoCodec() {}

    /** Default frames captured per second of video (Gemma-style low-fps sampling). */
    public static final int DEFAULT_FPS = 1;

    /** Default frame cap: each frame is ~256 image tokens, so 16 frames is already ~4k tokens. */
    public static final int DEFAULT_MAX_FRAMES = 16;

    /**
     * Sample a video into a {@link Media.Video} at {@link #DEFAULT_FPS}, capped at {@link
     * #DEFAULT_MAX_FRAMES}.
     */
    public static Media.Video load(Path path) throws IOException {
        return load(path, DEFAULT_FPS, DEFAULT_MAX_FRAMES);
    }

    /**
     * Sample {@code fps} frames per second, at most {@code maxFrames}, into a {@link Media.Video}
     * (audio null). The frames carry the video timeline: frame i is at {@code i / fps} seconds.
     */
    public static Media.Video load(Path path, int fps, int maxFrames) throws IOException {
        if (fps <= 0 || maxFrames <= 0)
            throw new IllegalArgumentException("fps and maxFrames must be positive");
        Path dir = Files.createTempDirectory("jinfer-video");
        try {
            runFfmpeg(
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    path.toString(),
                    "-vf",
                    "fps=" + fps,
                    "-frames:v",
                    String.valueOf(maxFrames),
                    dir.resolve("f%04d.png").toString());
            List<Path> pngs;
            try (Stream<Path> s = Files.list(dir)) {
                pngs =
                        s.filter(p -> p.getFileName().toString().endsWith(".png"))
                                .sorted(Comparator.comparing(Path::getFileName))
                                .toList();
            }
            if (pngs.isEmpty()) throw new IOException("ffmpeg extracted no frames from " + path);
            List<Media.Image> frames = new ArrayList<>(pngs.size());
            for (Path png : pngs) frames.add(ImageCodec.load(png));
            return new Media.Video(frames.toArray(new Media.Image[0]), fps, null);
        } finally {
            deleteRecursive(dir);
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
