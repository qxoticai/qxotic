package com.qxotic.jinfer.x.boundary.media;

import com.qxotic.jinfer.x.boundary.Media;
import java.io.IOException;
import java.nio.file.Path;

/**
 * Encoded image bytes to a decoded {@link com.qxotic.jinfer.x.boundary.Media.Image}. Two
 * implementations, selected by {@link ImageCodec}: ffmpeg (broad formats, works under native-image,
 * needs ffmpeg on PATH) and ImageIO (no external process, JVM only - its {@code IIORegistry} finds
 * codecs through ServiceLoader and reflection, which native-image cannot configure reliably).
 */
public interface ImageDecoder {

    /** Decode an image file into a raw {@link Media.Image} (RGB, [0,1], HWC). */
    Media.Image load(Path path) throws IOException;

    /** Decode encoded image bytes into a raw {@link Media.Image} (RGB, [0,1], HWC). */
    Media.Image decode(byte[] encoded) throws IOException;

    /** Short backend name, for logging/diagnostics ("ffmpeg" / "imageio"). */
    String name();
}
