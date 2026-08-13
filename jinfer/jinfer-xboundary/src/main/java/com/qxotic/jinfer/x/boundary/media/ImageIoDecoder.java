package com.qxotic.jinfer.x.boundary.media;

import com.qxotic.jinfer.x.boundary.Media;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.file.Path;
import javax.imageio.ImageIO;

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
        return fromBuffered(ImageIO.read(path.toFile()), path.toString());
    }

    @Override
    public Media.Image decode(byte[] encoded) throws IOException {
        return fromBuffered(
                ImageIO.read(new ByteArrayInputStream(encoded)), "<" + encoded.length + " bytes>");
    }

    private static Media.Image fromBuffered(BufferedImage bi, String src) throws IOException {
        if (bi == null) {
            throw new IOException(
                    "javax.imageio could not decode " + src + " (unsupported format?)");
        }
        int h = bi.getHeight(), w = bi.getWidth();
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
