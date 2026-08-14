package com.qxotic.jinfer.x.boundary.media;

import com.qxotic.jinfer.x.boundary.Media;
import java.io.IOException;
import java.nio.file.Path;

/**
 * Selects and caches an {@link ImageDecoder}: ffmpeg under native-image, ImageIO on a JVM,
 * overridden by {@code -Djinfer.imageDecoder=ffmpeg|imageio}.
 *
 * <p>ImageIO is loaded through a NON-CONSTANT class name on purpose (see {@link Codecs#reflect}) -
 * a constant one would let native-image fold the {@code Class.forName} and pull {@code
 * java.desktop} into the image.
 */
public final class ImageCodec {

    /** 4096 x 4096 RGB already expands to 192 MiB as the public float representation. */
    static final long MAX_PIXELS = 4096L * 4096;

    private ImageCodec() {}

    static void checkDimensions(int width, int height) throws IOException {
        long pixels = (long) width * height;
        if (width <= 0 || height <= 0 || pixels > MAX_PIXELS) {
            throw new IOException(
                    "image dimensions "
                            + width
                            + "x"
                            + height
                            + " exceed the "
                            + MAX_PIXELS
                            + "-pixel limit");
        }
    }

    private static volatile ImageDecoder decoder;

    /**
     * Decode an image file into a raw {@link Media.Image} (RGB, [0,1], HWC) via the selected
     * backend.
     */
    public static Media.Image load(Path path) throws IOException {
        return decoder().load(path);
    }

    /**
     * Decode encoded image bytes into a raw {@link Media.Image} (RGB, [0,1], HWC) via the selected
     * backend.
     */
    public static Media.Image decode(byte[] encoded) throws IOException {
        return decoder().decode(encoded);
    }

    /** The active decoder, lazily selected and cached. */
    public static ImageDecoder decoder() {
        ImageDecoder d = decoder;
        if (d == null) {
            synchronized (ImageCodec.class) {
                d = decoder;
                if (d == null) {
                    decoder = d = select();
                }
            }
        }
        return d;
    }

    private static ImageDecoder select() {
        String choice = Codecs.choice("jinfer.imageDecoder", "imageio");
        return switch (choice) {
            case "ffmpeg" -> new FfmpegImageDecoder();
            case "imageio" ->
                    // the explicit type witness: T must be ImageDecoder, not the fallback's
                    // concrete class, or the erased cast lands on the wrong type
                    Codecs.<ImageDecoder>reflect(
                            "com.qxotic.jinfer.x.boundary.media.ImageIoDecoder",
                            FfmpegImageDecoder::new);
            default ->
                    throw new IllegalArgumentException(
                            "unknown -Djinfer.imageDecoder='"
                                    + choice
                                    + "' (expected 'ffmpeg' or 'imageio')");
        };
    }
}
