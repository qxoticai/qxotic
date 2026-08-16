package com.qxotic.jinfer.boundary.media;

import com.qxotic.jinfer.boundary.Media;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
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
 * reader, keeping the RGB/[0,1]/HWC contract). Stateless; loud on failure.
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
        byte[] ppms = Ffmpeg.run(cmd, null);

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
                        Ffmpeg.run(cmd, null, Duration.ofSeconds(30), 64 << 10),
                        StandardCharsets.US_ASCII)
                .strip();
    }
}
