// Facade over the pluggable ImageDecoder. Selects the backend at runtime and caches it:
//   -Djinfer.imageDecoder=ffmpeg|imageio   (explicit override)
//   default: ffmpeg under GraalVM native-image (where javax.imageio is impractical), imageio on a
// normal JVM.
// ffmpeg is referenced directly (native-image-safe, always present). imageio is loaded REFLECTIVELY
// via a
// non-constant class name so native-image does not fold the Class.forName and pull java.desktop
// into the
// image; if it is requested but unavailable (e.g. inside a native image), it falls back to ffmpeg.
package com.qxotic.jinfer.media;

import com.qxotic.jinfer.Media;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Locale;

public final class ImageCodec {

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
            System.err.println(
                    "image decoder '"
                            + className
                            + "' unavailable ("
                            + e
                            + "); falling back to ffmpeg");
            return new FfmpegImageDecoder();
        }
    }
}
