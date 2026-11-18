package com.qxotic.jinfer.codecs;

import com.qxotic.jinfer.media.Media;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import javax.imageio.ImageIO;
import javax.imageio.ImageReader;
import javax.imageio.stream.ImageInputStream;

/**
 * The {@code javax.imageio} decoder, and the JVM default: no external process. Never referenced
 * statically - {@link ImageCodec} loads it reflectively so a native image does not pull in {@code
 * java.desktop}.
 */
public final class ImageIoDecoder implements ImageDecoder {

    @Override
    public String name() {
        return "imageio";
    }

    @Override
    public Media.Image load(Path path) throws IOException {
        try (ImageInputStream input = ImageIO.createImageInputStream(path.toFile())) {
            if (input == null) { // ImageIO's answer for any file it cannot open: say why instead
                if (Files.isDirectory(path)) throw new IOException(path + " is a directory");
                Files.newByteChannel(path).close(); // NoSuchFileException, AccessDeniedException
            }
            return read(input, path.toString());
        }
    }

    @Override
    public Media.Image decode(byte[] encoded) throws IOException {
        try (ImageInputStream input =
                ImageIO.createImageInputStream(new ByteArrayInputStream(encoded))) {
            return read(input, "<" + encoded.length + " bytes>");
        }
    }

    private static Media.Image read(ImageInputStream input, String src) throws IOException {
        if (input == null) throw unreadable(src);
        var readers = ImageIO.getImageReaders(input);
        if (!readers.hasNext()) throw unreadable(src);
        ImageReader reader = readers.next();
        try {
            reader.setInput(input, true, true);
            ImageCodec.checkDimensions(reader.getWidth(0), reader.getHeight(0));
            return fromBuffered(reader.read(0));
        } finally {
            reader.dispose();
        }
    }

    /** A format the JDK has no reader for: WEBP, HEIC and AVIF among them. */
    private static IOException unreadable(String src) {
        return new IOException(
                "javax.imageio cannot read "
                        + src
                        + ": it reads PNG, JPEG, GIF, BMP and TIFF;"
                        + " -Djinfer.imageDecoder=ffmpeg reads the rest");
    }

    private static Media.Image fromBuffered(BufferedImage bi) throws IOException {
        int h = bi.getHeight(), w = bi.getWidth();
        ImageCodec.checkDimensions(w, h);
        float[] v = new float[h * w * 3];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int rgb = bi.getRGB(x, y), idx = (y * w + x) * 3;
                v[idx] = ((rgb >> 16) & 0xff) / 255f;
                v[idx + 1] = ((rgb >> 8) & 0xff) / 255f;
                v[idx + 2] = (rgb & 0xff) / 255f;
            }
        }
        return new Media.Image(v, h, w, 3);
    }
}
