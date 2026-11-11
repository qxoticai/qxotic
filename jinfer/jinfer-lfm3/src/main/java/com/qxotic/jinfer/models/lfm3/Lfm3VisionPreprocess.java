package com.qxotic.jinfer.models.lfm3;

import com.qxotic.jinfer.Parallel;

/**
 * LFM3-VL image preprocessing as llama.cpp's {@code mtmd_image_preprocessor_lfm3vl} does it: the
 * transformers Qwen2-VL {@code smart_resize} to a multiple of {@code patch * merge}, then a stretch
 * (no letterbox) with torchvision's antialiased bicubic on 8-bit RGB ({@link
 * com.qxotic.jinfer.kernels.Resample#bicubic} with int16 weights), then {@code (v / 255 - mean) /
 * std}.
 */
final class Lfm3VisionPreprocess {

    private Lfm3VisionPreprocess() {}

    /**
     * {@code {width, height}}: each edge rounded half-to-even to the factor, then scaled into
     * {@code [minPixels, maxPixels]} keeping the aspect, in double as transformers computes it.
     */
    static int[] targetSize(int width, int height, int factor, long minPixels, long maxPixels) {
        double w = width, h = height;
        int wBar = (int) Math.rint(w / factor) * factor;
        int hBar = (int) Math.rint(h / factor) * factor;
        if ((long) wBar * hBar > maxPixels) {
            double beta = Math.sqrt(h * w / maxPixels);
            wBar = (int) Math.floor(w / beta / factor) * factor;
            hBar = (int) Math.floor(h / beta / factor) * factor;
        } else if ((long) wBar * hBar < minPixels) {
            double beta = Math.sqrt(minPixels / (h * w));
            wBar = (int) Math.ceil(w * beta / factor) * factor;
            hBar = (int) Math.ceil(h * beta / factor) * factor;
        }
        return new int[] {Math.max(factor, wBar), Math.max(factor, hBar)};
    }

    /**
     * {@code {width, height}} of every frame of a video of {@code frames} sampled frames, as the
     * transformers {@code Lfm3VlVideoProcessor} sizes them: the Qwen-VL video {@code smart_resize}
     * over its fixed budget of 4,096 to 25,165,824 pixels for the whole clip, two frames a temporal
     * patch.
     */
    static int[] videoSize(int frames, int width, int height, int factor) {
        final long minPixels = 4_096, maxPixels = 25_165_824;
        double w = width, h = height;
        if (h < factor || w < factor) { // an edge below the factor is scaled up to it first
            double scale = Math.max(factor / h, factor / w);
            h = (int) (h * scale);
            w = (int) (w * scale);
        }
        long tBar = (long) Math.rint(frames / 2.0) * 2;
        int wBar = (int) Math.rint(w / factor) * factor;
        int hBar = (int) Math.rint(h / factor) * factor;
        if (tBar * hBar * wBar > maxPixels) {
            double beta = Math.sqrt(frames * h * w / maxPixels);
            wBar = (int) Math.floor(w / beta / factor) * factor;
            hBar = (int) Math.floor(h / beta / factor) * factor;
        } else if (tBar * hBar * wBar < minPixels) {
            double beta = Math.sqrt(minPixels / (frames * h * w));
            wBar = (int) Math.ceil(w * beta / factor) * factor;
            hBar = (int) Math.ceil(h * beta / factor) * factor;
        }
        return new int[] {Math.max(factor, wBar), Math.max(factor, hBar)};
    }

    /**
     * Patch vectors {@code [patchesY * patchesX][3 * patch * patch]} in raster order, channel then
     * row then column within a patch (the conv kernel's layout), of the normalized pixels.
     */
    static float[] patches(int[] rgb, int width, int height, int patch, float[] mean, float[] std) {
        int patchesX = width / patch, patchesY = height / patch, area = patch * patch;
        float[] out = new float[patchesX * patchesY * 3 * area];
        Parallel.forLoop(
                patchesY,
                py -> {
                    for (int px = 0; px < patchesX; px++) {
                        int at = (py * patchesX + px) * 3 * area;
                        for (int c = 0; c < 3; c++)
                            for (int y = 0; y < patch; y++)
                                for (int x = 0; x < patch; x++) {
                                    int pixel = (py * patch + y) * width + px * patch + x;
                                    out[at++] = (rgb[pixel * 3 + c] / 255f - mean[c]) / std[c];
                                }
                    }
                });
        return out;
    }
}
