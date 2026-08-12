package com.qxotic.jinfer.x.models.gemma4;

import com.qxotic.jinfer.x.PanamaMemoryArena;
import com.qxotic.jinfer.x.Parallel;
import com.qxotic.jinfer.x.Views;
import com.qxotic.jinfer.x.boundary.Media;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;

/** Shared Gemma 4 aspect-preserving resize and patch extraction. */
final class VisionPreprocess {
    private VisionPreprocess() {}

    static final boolean SMART_RESIZE = !Boolean.getBoolean("vis.squareResize");
    static final int IMAGE_TOKEN_BUDGET = validatedBudget("jinfer.gemma4.imageTokenBudget", -1);
    static final int VIDEO_TOKEN_BUDGET = validatedBudget("jinfer.gemma4.videoTokenBudget", 70);

    private static int validatedBudget(String property, int defaultValue) {
        String value = System.getProperty(property);
        if (value == null) return defaultValue;
        int budget = Integer.parseInt(value.trim());
        if (budget != 70 && budget != 140 && budget != 280 && budget != 560 && budget != 1120)
            throw new IllegalArgumentException(
                    property + " must be 70|140|280|560|1120, got " + budget);
        return budget;
    }

    static int budget(int defaultValue) {
        return IMAGE_TOKEN_BUDGET > 0 ? IMAGE_TOKEN_BUDGET : defaultValue;
    }

    static int[] smartResize(int width, int height, int factor, int minPixels, int maxPixels) {
        if (width <= 0 || height <= 0 || factor <= 0 || minPixels <= 0 || maxPixels < minPixels)
            throw new IllegalArgumentException("invalid smart-resize geometry");
        int resizedWidth = Math.max(factor, Math.round((float) width / factor) * factor);
        int resizedHeight = Math.max(factor, Math.round((float) height / factor) * factor);
        long area = (long) resizedWidth * resizedHeight;
        if (area > maxPixels) {
            double beta = Math.sqrt((double) width * height / maxPixels);
            resizedWidth = Math.max(factor, (int) (Math.floor(width / beta / factor) * factor));
            resizedHeight = Math.max(factor, (int) (Math.floor(height / beta / factor) * factor));
        } else if (area < minPixels) {
            double beta = Math.sqrt((double) minPixels / ((double) width * height));
            resizedWidth = (int) (Math.ceil(width * beta / factor) * factor);
            resizedHeight = (int) (Math.ceil(height * beta / factor) * factor);
        }
        return new int[] {resizedWidth, resizedHeight};
    }

    static float[] toCHW(Media.Image image, int targetWidth, int targetHeight) {
        if (targetWidth <= 0 || targetHeight <= 0)
            throw new IllegalArgumentException("target dimensions must be positive");
        int plane = Math.multiplyExact(targetHeight, targetWidth);
        float[] out = new float[Math.multiplyExact(3, plane)];
        int height = image.height(), width = image.width(), channels = image.channels();
        float[] values = image.values();
        for (int y = 0; y < targetHeight; y++)
            for (int x = 0; x < targetWidth; x++) {
                float fy = (y + 0.5f) * height / targetHeight - 0.5f;
                float fx = (x + 0.5f) * width / targetWidth - 0.5f;
                int y0 = Math.max(0, Math.min(height - 1, (int) Math.floor(fy)));
                int y1 = Math.min(height - 1, y0 + 1);
                int x0 = Math.max(0, Math.min(width - 1, (int) Math.floor(fx)));
                int x1 = Math.min(width - 1, x0 + 1);
                float wy = Math.max(0f, fy - y0), wx = Math.max(0f, fx - x0);
                for (int c = 0; c < 3; c++) {
                    int cc = Math.min(c, channels - 1);
                    float a = values[(y0 * width + x0) * channels + cc];
                    float b = values[(y0 * width + x1) * channels + cc];
                    float d0 = values[(y1 * width + x0) * channels + cc];
                    float d1 = values[(y1 * width + x1) * channels + cc];
                    out[c * plane + y * targetWidth + x] =
                            a * (1 - wx) * (1 - wy)
                                    + b * wx * (1 - wy)
                                    + d0 * (1 - wx) * wy
                                    + d1 * wx * wy;
                }
            }
        return out;
    }

    static MemoryView<MemorySegment> im2col(
            Media.Image image,
            int targetWidth,
            int targetHeight,
            int patchSize,
            PanamaMemoryArena scratch) {
        if (patchSize <= 0 || targetWidth % patchSize != 0 || targetHeight % patchSize != 0)
            throw new IllegalArgumentException("target dimensions must be divisible by patchSize");
        int patchesX = targetWidth / patchSize, patchesY = targetHeight / patchSize;
        int count = Math.multiplyExact(patchesX, patchesY);
        int patchVector = Math.multiplyExact(3, Math.multiplyExact(patchSize, patchSize));
        int plane = Math.multiplyExact(targetHeight, targetWidth);
        float[] chw = toCHW(image, targetWidth, targetHeight);
        // Gather into a heap buffer first (parallel-safe), then ONE bulk copy on the owning
        // thread: a checked copy inside forRows would trip thread-confinement on confined arenas.
        float[] data = new float[Math.multiplyExact(count, patchVector)];
        Parallel.forRows(
                count,
                patch -> {
                    int gy = patch / patchesX, gx = patch % patchesX, column = patch * patchVector;
                    for (int c = 0; c < 3; c++)
                        for (int ky = 0; ky < patchSize; ky++)
                            for (int kx = 0; kx < patchSize; kx++) {
                                data[column++] =
                                        chw[
                                                                c * plane
                                                                        + (gy * patchSize + ky)
                                                                                * targetWidth
                                                                        + gx * patchSize
                                                                        + kx]
                                                        * 2f
                                                - 1f;
                            }
                });
        MemoryView<MemorySegment> flat = Views.allocateF32(scratch, count, patchVector);
        Views.copyFromArray(flat, 0, data, 0, data.length, "patches");
        return flat;
    }
}
