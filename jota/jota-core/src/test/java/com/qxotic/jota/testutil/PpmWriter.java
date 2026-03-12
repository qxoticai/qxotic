package com.qxotic.jota.testutil;

import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

public final class PpmWriter {

    private PpmWriter() {}

    public static void write(Path path, int width, int height, byte[] rgb) throws IOException {
        Path parent = path.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        try (OutputStream out = new BufferedOutputStream(Files.newOutputStream(path))) {
            out.write(
                    ("P6\n" + width + " " + height + "\n255\n")
                            .getBytes(StandardCharsets.US_ASCII));
            out.write(rgb);
        }
    }

    public static void writeP6(Path path, int width, int height, byte[] rgb) throws IOException {
        write(path, width, height, rgb);
    }

    public static void write(Path path, int width, int height, float[] r, float[] g, float[] b)
            throws IOException {
        byte[] rgb = new byte[width * height * 3];
        for (int i = 0; i < width * height; i++) {
            rgb[i * 3] = (byte) toByte(r[i]);
            rgb[i * 3 + 1] = (byte) toByte(g[i]);
            rgb[i * 3 + 2] = (byte) toByte(b[i]);
        }
        write(path, width, height, rgb);
    }

    private static int toByte(float v) {
        float clamped = Math.max(0.0f, Math.min(1.0f, v));
        return Math.round(clamped * 255.0f);
    }
}
