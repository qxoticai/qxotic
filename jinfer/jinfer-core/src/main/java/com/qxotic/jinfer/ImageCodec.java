// Native-image-safe image codec. Shells out to ffmpeg to decode PNG/JPEG/WebP/... into raw RGB, producing
// a generic Media.Image (values in [0,1], HWC-interleaved, 3 channels R,G,B) - the exact convention the old
// javax.imageio loader used, so the vision encoder output is unchanged.
//
// Why ffmpeg and not javax.imageio: ImageIO's IIORegistry uses ServiceLoader + reflection to discover codec
// plugins, which needs build-time configuration and is fragile under GraalVM native-image. An external ffmpeg
// process has none of that (and decodes far more formats). This mirrors the audio path (ffmpeg mp3 -> wav).
//
// DECODE ONLY: ffmpeg does format -> raw RGB. The model-specific smart_resize stays in the vision module
// (VisionPreprocess), so Media.Image remains raw, model-agnostic pixels.
package com.qxotic.jinfer;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;

public final class ImageCodec {

    private ImageCodec() {
    }

    /** Decode an image file (png/jpeg/webp/gif/...) to a {@link Media.Image}: RGB, values in [0,1], HWC. */
    public static Media.Image load(Path path) throws IOException {
        return parsePpm(runFfmpeg(new String[]{"ffmpeg", "-hide_banner", "-loglevel", "error",
                "-i", path.toString(), "-f", "image2pipe", "-vcodec", "ppm", "-pix_fmt", "rgb24", "-"}, null));
    }

    /** Decode encoded image bytes (png/jpeg/...) to a {@link Media.Image}: RGB, values in [0,1], HWC. */
    public static Media.Image decode(byte[] encoded) throws IOException {
        return parsePpm(runFfmpeg(new String[]{"ffmpeg", "-hide_banner", "-loglevel", "error",
                "-i", "pipe:0", "-f", "image2pipe", "-vcodec", "ppm", "-pix_fmt", "rgb24", "-"}, encoded));
    }

    /** Run ffmpeg with the given argv, optionally feeding {@code stdin}, returning stdout (the PPM). */
    private static byte[] runFfmpeg(String[] cmd, byte[] stdin) throws IOException {
        ProcessBuilder pb = new ProcessBuilder(cmd).redirectError(ProcessBuilder.Redirect.INHERIT);
        Process p;
        try {
            p = pb.start();
        } catch (IOException e) {
            throw new IOException("failed to launch ffmpeg (is it on PATH?): " + e.getMessage(), e);
        }
        // Feed stdin on a daemon thread to avoid deadlock while we drain stdout on this thread.
        if (stdin != null) {
            Thread feeder = new Thread(() -> {
                try (var os = p.getOutputStream()) {
                    os.write(stdin);
                } catch (IOException ignored) {
                    // ffmpeg closed the pipe early (e.g. it errored); the exit code below reports it.
                }
            }, "ffmpeg-stdin");
            feeder.setDaemon(true);
            feeder.start();
        } else {
            p.getOutputStream().close();
        }
        byte[] out;
        try (InputStream is = p.getInputStream()) {
            out = is.readAllBytes();
        }
        int code;
        try {
            code = p.waitFor();
        } catch (InterruptedException e) {
            p.destroyForcibly();
            Thread.currentThread().interrupt();
            throw new IOException("interrupted while decoding image", e);
        }
        if (code != 0) {
            throw new IOException("ffmpeg exited " + code + " (unsupported/corrupt image, or ffmpeg missing)");
        }
        return out;
    }

    /** Parse a binary PPM (P6): magic "P6", then width, height, maxval as whitespace-separated ASCII (with
     *  optional '#' comment lines), a single whitespace byte, then width*height*3 raw RGB bytes (row-major,
     *  interleaved). Values scaled to [0,1], HWC - identical to the former ImageIO loader. */
    static Media.Image parsePpm(byte[] ppm) throws IOException {
        int[] pos = {0};
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
        int base = pos[0];                 // token() consumed exactly one whitespace after maxval
        int need = Math.multiplyExact(Math.multiplyExact(w, h), 3);
        if (base + need > ppm.length) {
            throw new IOException("truncated PPM: need " + need + " pixel bytes, have " + (ppm.length - base));
        }
        float[] v = new float[need];
        for (int i = 0; i < need; i++) {
            v[i] = (ppm[base + i] & 0xff) / 255f;
        }
        return new Media.Image(v, h, w, 3);
    }

    private static int parseInt(String s, String field) throws IOException {
        try {
            return Integer.parseInt(s);
        } catch (NumberFormatException e) {
            throw new IOException("malformed PPM " + field + ": '" + s + "'");
        }
    }

    /** Read the next whitespace-delimited token, skipping leading whitespace and '#' comment lines, and
     *  advancing past the single whitespace byte that terminates the token (as PPM requires before the
     *  binary raster). */
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
        pos[0] = i + 1;   // consume exactly one terminating whitespace byte
        return tok;
    }
}
