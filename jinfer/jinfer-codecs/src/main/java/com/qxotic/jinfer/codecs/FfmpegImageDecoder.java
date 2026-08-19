package com.qxotic.jinfer.codecs;

import com.qxotic.jinfer.media.Media;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.List;

/**
 * The ffmpeg image decoder: broad format support, native-image-safe, and portable to wherever
 * ffmpeg is on PATH. DECODE ONLY - model-specific resizing belongs to the vision port.
 */
public final class FfmpegImageDecoder implements ImageDecoder {

    @Override
    public String name() {
        return "ffmpeg";
    }

    @Override
    public Media.Image load(Path path) throws IOException {
        return parsePpm(Ffmpeg.run(ffmpegArgs(path.toString()), null));
    }

    @Override
    public Media.Image decode(byte[] encoded) throws IOException {
        return parsePpm(Ffmpeg.run(ffmpegArgs("pipe:0"), encoded));
    }

    private static List<String> ffmpegArgs(String input) {
        return List.of(
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                input,
                "-f",
                "image2pipe",
                "-vcodec",
                "ppm",
                "-pix_fmt",
                "rgb24",
                "-");
    }

    /**
     * Parse a binary PPM (P6): magic "P6", then width, height, maxval as whitespace-separated ASCII
     * (with optional '#' comment lines), a single whitespace byte, then width*height*3 raw RGB
     * bytes (row-major, interleaved). Values scaled to [0,1], HWC - identical to the javax.imageio
     * loader.
     */
    static Media.Image parsePpm(byte[] ppm) throws IOException {
        return parsePpm(ppm, new int[] {0});
    }

    /**
     * Parse one PPM starting at {@code pos[0]}, advancing {@code pos} past its raster - the unit of
     * a multi-frame {@code image2pipe} stream (each frame is a self-describing P6, so a video's
     * frames carry their own dimensions; no probe, no per-frame process).
     */
    static Media.Image parsePpm(byte[] ppm, int[] pos) throws IOException {
        String magic = token(ppm, pos);
        if (!"P6".equals(magic)) {
            throw new IOException("expected a P6 PPM from ffmpeg, got '" + magic + "'");
        }
        int w = parseInt(token(ppm, pos), "width");
        int h = parseInt(token(ppm, pos), "height");
        int maxval = parseInt(token(ppm, pos), "maxval");
        if (maxval != 255) {
            throw new IOException("expected 8-bit PPM (maxval 255), got " + maxval);
        }
        int base = pos[0]; // token() consumed exactly one whitespace after maxval
        ImageCodec.checkDimensions(w, h);
        int need;
        try {
            need = Math.multiplyExact(Math.multiplyExact(w, h), 3);
        } catch (ArithmeticException e) {
            throw new IOException("PPM dimensions are too large: " + w + "x" + h, e);
        }
        if (base + need > ppm.length) {
            throw new IOException(
                    "truncated PPM: need " + need + " pixel bytes, have " + (ppm.length - base));
        }
        float[] v = new float[need];
        for (int i = 0; i < need; i++) {
            v[i] = (ppm[base + i] & 0xff) / 255f;
        }
        pos[0] = base + need;
        return new Media.Image(v, h, w, 3);
    }

    private static int parseInt(String s, String field) throws IOException {
        try {
            return Integer.parseInt(s);
        } catch (NumberFormatException e) {
            throw new IOException("malformed PPM " + field + ": '" + s + "'");
        }
    }

    /**
     * Read the next whitespace-delimited token, skipping leading whitespace and '#' comment lines,
     * and advancing past the single whitespace byte that terminates the token (as PPM requires
     * before the binary raster).
     */
    private static String token(byte[] b, int[] pos) throws IOException {
        int i = pos[0];
        while (i < b.length) {
            int c = b[i] & 0xff;
            if (c == '#') {
                while (i < b.length && b[i] != '\n') i++;
            } else if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                i++;
            } else {
                break;
            }
        }
        int start = i;
        while (i < b.length) {
            int c = b[i] & 0xff;
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') break;
            i++;
        }
        if (i == start) {
            throw new IOException("malformed PPM header (unexpected end)");
        }
        String tok = new String(b, start, i - start, StandardCharsets.US_ASCII);
        pos[0] = i + 1; // consume exactly one terminating whitespace byte
        return tok;
    }
}
