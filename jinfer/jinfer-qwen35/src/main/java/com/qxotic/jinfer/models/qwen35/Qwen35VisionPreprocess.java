package com.qxotic.jinfer.models.qwen35;

import com.qxotic.jinfer.kernels.Resample;
import com.qxotic.jinfer.media.Media;

/**
 * Qwen3-VL image sizing and normalization, as the Hugging Face processor and llama.cpp's qwen3vl
 * preprocessor do it: the image's merged-token count kept in {@code [8, 4096]} by default (or the
 * mmproj's own pixel bounds), each edge rounded half to even to a multiple of {@code patch_size *
 * spatial_merge_size}, then a bicubic stretch to that size (no letterbox).
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

    /**
     * Multiple-of-{@code align} smart resize keeping the projected block count inside {@code
     * [minTokens, maxTokens]}. The math is llama.cpp's {@code smart_resize}, so changing it changes
     * cache identity.
     */
    static int[] smartResize(int width, int height, int align, int minTokens, int maxTokens) {
        if (width <= 0 || height <= 0 || align <= 0 || minTokens <= 0 || maxTokens < minTokens)
            throw new IllegalArgumentException("invalid smart-resize geometry");
        // half to even, in double: llama.cpp's nearbyint, transformers' round (2000/32 = 62.5 is
        // 62)
        int wBar = Math.max(align, (int) Math.rint((double) width / align) * align);
        int hBar = Math.max(align, (int) Math.rint((double) height / align) * align);
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
     * The image stretched to {@code targetWidth x targetHeight} with Pillow's bicubic on 8-bit RGB,
     * as the Hugging Face processor and llama.cpp's qwen3vl preprocessor do (no letterbox), then
     * {@code (v / 255 - mean) / std} per channel in CHW plane order.
     */
    static float[] normalize(
            Media.Image image, int targetWidth, int targetHeight, float[] mean, float[] std) {
        int plane = Math.multiplyExact(targetHeight, targetWidth);
        int[] rgb = Resample.bicubic(image, targetWidth, targetHeight, false);
        float[] out = new float[Math.multiplyExact(3, plane)];
        for (int c = 0; c < 3; c++)
            for (int p = 0; p < plane; p++)
                out[c * plane + p] = (rgb[p * 3 + c] / 255f - mean[c]) / std[c];
        return out;
    }
}
