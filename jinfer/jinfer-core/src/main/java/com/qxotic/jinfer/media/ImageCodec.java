package com.qxotic.jinfer.media;

import com.qxotic.jinfer.Media;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Locale;

/**
 * Selects and caches an {@link ImageDecoder}: ffmpeg under native-image, ImageIO on a JVM,
 * overridden by {@code -Djinfer.imageDecoder=ffmpeg|imageio}.
 *
 * <p>ImageIO is loaded through a NON-CONSTANT class name on purpose - a constant one would let
 * native-image fold the {@code Class.forName} and pull {@code java.desktop} into the image.
 */
public final class ImageCodec {

    private static final System.Logger LOG = System.getLogger("jinfer.media");

    private ImageCodec() {}

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
        String choice = System.getProperty("jinfer.imageDecoder");
        if (choice == null || choice.isBlank()) {
            boolean nativeImage = System.getProperty("org.graalvm.nativeimage.imagecode") != null;
            choice = nativeImage ? "ffmpeg" : "imageio";
        }
        return switch (choice.toLowerCase(Locale.ROOT)) {
            case "ffmpeg" -> new FfmpegImageDecoder();
            case "imageio" -> loadReflectively("com.qxotic.jinfer.media.ImageIoDecoder");
            default ->
                    throw new IllegalArgumentException(
                            "unknown -Djinfer.imageDecoder='"
                                    + choice
                                    + "' (expected 'ffmpeg' or 'imageio')");
        };
    }

    /**
     * Instantiate a decoder by name via reflection. Passing the name as an argument (not a literal
     * at the Class.forName site) keeps native-image from constant-folding it, so the ImageIO
     * backend and java.desktop stay out of native images. Falls back to ffmpeg if the backend can't
     * load.
     */
    private static ImageDecoder loadReflectively(String className) {
        try {
            return (ImageDecoder) Class.forName(className).getDeclaredConstructor().newInstance();
        } catch (ReflectiveOperationException | LinkageError e) {
            LOG.log(
                    System.Logger.Level.WARNING,
                    "image decoder ''{0}'' unavailable ({1}); falling back to ffmpeg",
                    className,
                    e);
            return new FfmpegImageDecoder();
        }
    }
}
