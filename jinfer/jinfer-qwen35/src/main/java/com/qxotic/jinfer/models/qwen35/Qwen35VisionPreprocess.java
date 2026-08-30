package com.qxotic.jinfer.models.qwen35;

import com.qxotic.jinfer.media.Media;

/**
 * Qwen3-VL image sizing and normalization. llama.cpp's qwen3vl preprocessor caps the projected
 * block count to {@code [8, 4096]} merged tokens by default (min/max image tokens), resizes to a
 * multiple of {@code patch_size * spatial_merge_size}, then center-fits into the target with black
 * padding ({@code PAD_CEIL}). These helpers are the pixel-level contract the tower consumes.
 */
final class Qwen35VisionPreprocess {
    private Qwen35VisionPreprocess() {}

    static final int MIN_IMAGE_TOKENS = validated("jinfer.qwen35.imageMinTokens", 8);
    static final int MAX_IMAGE_TOKENS = validated("jinfer.qwen35.imageMaxTokens", 4096);

    private static int validated(String property, int defaultValue) {
        String value = System.getProperty(property);
        if (value == null) return defaultValue;
        int budget;
        try {
            budget = Integer.parseInt(value.trim());
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(
                    property + " must be an integer, got '" + value + "'");
        }
        if (budget <= 0)
            throw new IllegalArgumentException(property + " must be positive, got " + budget);
        return budget;
    }

    /** Number of merged (post-spatial-merge) rows an image produces. */
    static int positions(Media.Image image, int patchSize, int merge) {
        int[] size =
                smartResize(
                        image.width(),
                        image.height(),
                        Math.multiplyExact(patchSize, merge),
                        MIN_IMAGE_TOKENS,
                        MAX_IMAGE_TOKENS);
        int patchesX = size[0] / patchSize, patchesY = size[1] / patchSize;
        return Math.multiplyExact(patchesX, patchesY) / Math.multiplyExact(merge, merge);
    }

    /**
     * Multiple-of-{@code align} smart resize keeping the projected block count inside {@code
     * [minTokens, maxTokens]}. The math is llama.cpp's {@code smart_resize} (align-corners family),
     * so changing it changes cache identity.
     */
    static int[] smartResize(int width, int height, int align, int minTokens, int maxTokens) {
        if (width <= 0 || height <= 0 || align <= 0 || minTokens <= 0 || maxTokens < minTokens)
            throw new IllegalArgumentException("invalid smart-resize geometry");
        int wBar = Math.max(align, Math.round((float) width / align) * align);
        int hBar = Math.max(align, Math.round((float) height / align) * align);
        long minPixels = (long) minTokens * align * align;
        long maxPixels = (long) maxTokens * align * align;
        if ((long) wBar * hBar > maxPixels) {
            double beta = Math.sqrt((double) width * height / maxPixels);
            wBar = Math.max(align, (int) (Math.floor(width / beta / align) * align));
            hBar = Math.max(align, (int) (Math.floor(height / beta / align) * align));
        } else if ((long) wBar * hBar < minPixels) {
            double beta = Math.sqrt((double) minPixels / ((double) width * height));
            wBar = Math.max(align, (int) (Math.ceil(width * beta / align) * align));
            hBar = Math.max(align, (int) (Math.ceil(height * beta / align) * align));
        }
        return new int[] {wBar, hBar};
    }

    /**
     * Align-corners bilinear resize to {@code targetWidth x targetHeight}, normalized to [-1,1]
     * (mean = std = 0.5) in CHW plane order.
     *
     * <p>llama.cpp's qwen3vl preprocessor uses PAD_CEIL (scale-to-fit, center, fill the remainder
     * black), not scale-to-fill. The black pad is applied in [0,1] value space before the {@code
     * 2*v-1} normalization.
     */
    static float[] normalize(Media.Image image, int targetWidth, int targetHeight) {
        return normalize(
                image,
                targetWidth,
                targetHeight,
                new float[] {0.5f, 0.5f, 0.5f},
                new float[] {0.5f, 0.5f, 0.5f});
    }

    static float[] normalize(
            Media.Image image, int targetWidth, int targetHeight, float[] mean, float[] std) {
        int plane = Math.multiplyExact(targetHeight, targetWidth);
        float[] out = new float[Math.multiplyExact(3, plane)];
        int height = image.height(), width = image.width(), channels = image.channels();
        float[] values = image.values();
        float scale = Math.min((float) targetWidth / width, (float) targetHeight / height);
        int newWidth = Math.min((int) Math.ceil(width * scale), targetWidth);
        int newHeight = Math.min((int) Math.ceil(height * scale), targetHeight);
        int offX = (targetWidth - newWidth) / 2;
        int offY = (targetHeight - newHeight) / 2;
        for (int c = 0; c < 3; c++) {
            int cc = Math.min(c, channels - 1);
            for (int y = 0; y < targetHeight; y++) {
                int srcY = y - offY;
                for (int x = 0; x < targetWidth; x++) {
                    int srcX = x - offX;
                    float v;
                    if (srcX < 0 || srcX >= newWidth || srcY < 0 || srcY >= newHeight) {
                        v = 0f; // black pad in [0,1] value space -> -1 after normalize
                    } else {
                        float fx = newWidth > 1 ? (float) srcX * (width - 1) / (newWidth - 1) : 0f;
                        float fy =
                                newHeight > 1 ? (float) srcY * (height - 1) / (newHeight - 1) : 0f;
                        int x0 = Math.min((int) fx, width - 1);
                        int x1 = Math.min(x0 + 1, width - 1);
                        int y0 = Math.min((int) fy, height - 1);
                        int y1 = Math.min(y0 + 1, height - 1);
                        float wx = fx - x0, wy = fy - y0;
                        float a = values[(y0 * width + x0) * channels + cc];
                        float b = values[(y0 * width + x1) * channels + cc];
                        float d0 = values[(y1 * width + x0) * channels + cc];
                        float d1 = values[(y1 * width + x1) * channels + cc];
                        v =
                                a * (1 - wx) * (1 - wy)
                                        + b * wx * (1 - wy)
                                        + d0 * (1 - wx) * wy
                                        + d1 * wx * wy;
                    }
                    out[c * plane + y * targetWidth + x] = (v - mean[c]) / std[c];
                }
            }
        }
        return out;
    }
}
