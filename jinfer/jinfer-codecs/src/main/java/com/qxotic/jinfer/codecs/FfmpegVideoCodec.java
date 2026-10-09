package com.qxotic.jinfer.codecs;

import com.qxotic.jinfer.media.Media;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.FileTime;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * The {@link VideoCodec} primitives via ffmpeg/ffprobe on PATH: {@code totalDuration} and {@code
 * framePeriod} probe the container ({@code -of csv=p=0}), and {@code framesAt} extracts ALL frames
 * in one ffmpeg process - one seeked input per timestamp ({@code -ss} before {@code -i}:
 * keyframe-fast, exact enough for frame sampling), trimmed to a frame each and concatenated to a
 * single multi-frame PPM stream on stdout (frames parse through {@link FfmpegImageDecoder}'s PPM
 * reader, keeping the RGB/[0,1]/HWC contract). Loud on failure. Both properties come from one
 * ffprobe run, remembered for the last file probed: every sampling policy asks for the two back to
 * back, and a second run would cost one more process per video.
 */
public final class FfmpegVideoCodec implements VideoCodec {

    static final FfmpegVideoCodec INSTANCE = new FfmpegVideoCodec();

    /** A file as probed: a rewrite changes its size or modification time. */
    private record Key(Path video, long size, FileTime modified) {}

    private record Probe(Key key, Duration duration, Duration framePeriod) {}

    private volatile Probe last;

    @Override
    public String name() {
        return "ffmpeg";
    }

    @Override
    public Duration totalDuration(Path video) throws IOException {
        return probe(video).duration();
    }

    @Override
    public Duration framePeriod(Path video) throws IOException {
        return probe(video).framePeriod();
    }

    private Probe probe(Path video) throws IOException {
        Key key = new Key(video, Files.size(video), Files.getLastModifiedTime(video));
        Probe cached = last;
        if (cached != null && cached.key().equals(key)) return cached;
        // one run, two lines: the video stream's r_frame_rate, then the container's duration
        String[] out =
                probe(
                                video,
                                "-select_streams",
                                "v:0",
                                "-show_entries",
                                "format=duration:stream=r_frame_rate")
                        .split("\\R");
        if (out.length != 2)
            throw new IOException(
                    "unexpected ffprobe output for " + video + ": " + String.join(" | ", out));
        Probe probe = new Probe(key, duration(video, out[1]), period(video, out[0]));
        last = probe;
        return probe;
    }

    private static Duration duration(Path video, String out) throws IOException {
        try {
            double d = Double.parseDouble(out);
            if (!(d > 0)) throw new IOException("ffprobe reported non-positive duration: " + out);
            return Duration.ofNanos((long) (d * 1e9));
        } catch (NumberFormatException e) {
            throw new IOException("unparsable ffprobe duration for " + video + ": " + out);
        }
    }

    private static Duration period(Path video, String out) throws IOException {
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
        int n = timestamps.length;
        if (n == 0) throw new IllegalArgumentException("no timestamps");
        // ONE process for all frames: one seeked input per timestamp (input-side -ss stays
        // keyframe-fast; the file is opened n times, never decoded end to end), each trimmed to
        // its first frame and concatenated into a single multi-frame PPM stream on stdout -
        // no temp files, no PNG codec, no per-frame process spawn (measured 121 ms/frame as
        // spawn+seek+png; the pipe leaves only the seek+decode).
        List<String> cmd = new ArrayList<>(List.of("ffmpeg", "-hide_banner", "-loglevel", "error"));
        StringBuilder fc = new StringBuilder();
        for (int k = 0; k < n; k++) {
            cmd.add("-ss");
            cmd.add(seekTime(timestamps[k]));
            cmd.add("-i");
            cmd.add(video.toString());
            fc.append('[').append(k).append(":v]trim=end_frame=1[f").append(k).append("];");
        }
        for (int k = 0; k < n; k++) fc.append("[f").append(k).append(']');
        // setpts=N + passthrough fps: seeked segments all restart near PTS 0, and the muxer's
        // default frame-rate sync DROPS colliding frames (measured 19 of 32 without this) -
        // renumbering gives monotonic timestamps and passthrough keeps every frame.
        fc.append("concat=n=").append(n).append(":v=1:a=0[c];[c]setpts=N[out]");
        cmd.addAll(
                List.of(
                        "-filter_complex",
                        fc.toString(),
                        "-map",
                        "[out]",
                        "-fps_mode",
                        "passthrough",
                        "-f",
                        "image2pipe",
                        "-vcodec",
                        "ppm",
                        "-pix_fmt",
                        "rgb24",
                        "-"));
        byte[] ppms = Subprocess.run(cmd, null);

        // Frames pair with timestamps by ORDER, so a short stream must throw, never mis-pair:
        // a timestamp past the final frame yields an empty segment and fewer than n frames.
        List<Media.Video.Frame> frames = new ArrayList<>(n);
        int[] pos = {0};
        for (int k = 0; k < n; k++) {
            if (pos[0] >= ppms.length) {
                throw new IOException(
                        "ffmpeg produced "
                                + k
                                + " of "
                                + n
                                + " frames from "
                                + video
                                + " (a timestamp seeks past the final frame?)");
            }
            frames.add(
                    new Media.Video.Frame(FfmpegImageDecoder.parsePpm(ppms, pos), timestamps[k]));
        }
        return new Media.Video(frames);
    }

    /** Millisecond ffmpeg seek, floored so a final-frame timestamp is never rounded past EOF. */
    static String seekTime(Duration timestamp) {
        return String.format(Locale.ROOT, "%.3f", timestamp.toMillis() / 1000d);
    }

    /** One ffprobe field as text ({@code -of csv=p=0}); loud on failure. */
    private static String probe(Path video, String... entries) throws IOException {
        List<String> cmd = new ArrayList<>(List.of("ffprobe", "-v", "error"));
        cmd.addAll(List.of(entries));
        cmd.addAll(List.of("-of", "csv=p=0", video.toString()));
        // US_ASCII, not the platform default: csv=p=0 output is digits and dots, and the default
        // charset differs across Linux/macOS/Windows.
        return new String(
                        Subprocess.run(cmd, null, Duration.ofSeconds(30), 64 << 10),
                        StandardCharsets.US_ASCII)
                .strip();
    }
}
