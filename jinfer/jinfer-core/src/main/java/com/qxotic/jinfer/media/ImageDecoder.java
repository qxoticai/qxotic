// Pluggable image decoder: encoded image bytes (PNG/JPEG/WebP/...) -> raw Media.Image (RGB, values in
// [0,1], HWC-interleaved, 3 channels). Two implementations:
//   - FfmpegImageDecoder: shells out to ffmpeg; native-image-safe, cross-platform (macOS/Windows/Linux),
//     broad format support; needs ffmpeg on PATH.
//   - ImageIoDecoder: javax.imageio; JVM-only (its IIORegistry ServiceLoader/reflection is fragile under
//     GraalVM native-image), no external process.
// Select via ImageCodec (-Djinfer.imageDecoder=ffmpeg|imageio; default ffmpeg under native-image, imageio on JVM).
package com.qxotic.jinfer.media;

import com.qxotic.jinfer.Media;

import java.io.IOException;
import java.nio.file.Path;

public interface ImageDecoder {

    /** Decode an image file into a raw {@link Media.Image} (RGB, [0,1], HWC). */
    Media.Image load(Path path) throws IOException;

    /** Decode encoded image bytes into a raw {@link Media.Image} (RGB, [0,1], HWC). */
    Media.Image decode(byte[] encoded) throws IOException;

    /** Short backend name, for logging/diagnostics ("ffmpeg" / "imageio"). */
    String name();
}
