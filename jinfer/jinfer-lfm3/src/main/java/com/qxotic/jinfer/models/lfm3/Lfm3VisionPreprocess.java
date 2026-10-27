package com.qxotic.jinfer.models.lfm3;

import com.qxotic.jinfer.Parallel;
import com.qxotic.jinfer.media.Media;

/**
 * LFM3-VL image preprocessing as llama.cpp's {@code mtmd_image_preprocessor_lfm3vl} does it: the
 * transformers Qwen2-VL {@code smart_resize} to a multiple of {@code patch * merge}, then a stretch
 * (no letterbox) with torchvision's antialiased bicubic on 8-bit RGB - Pillow's separable filter
 * with int16 weights - then {@code (v / 255 - mean) / std}.
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
     * The image as 8-bit RGB {@code [height][width][3]} resized to the target: alpha dropped and
     * grey replicated, as stb_image loads it.
     */
    static int[] resize(Media.Image image, int targetWidth, int targetHeight) {
        int width = image.width(), height = image.height(), channels = image.channels();
        float[] values = image.values();
        int[] rgb = new int[Math.multiplyExact(width * height, 3)];
        for (int p = 0; p < width * height; p++)
            for (int c = 0; c < 3; c++)
                rgb[p * 3 + c] =
                        Math.round(values[p * channels + Math.min(c, channels - 1)] * 255f);
        if (targetWidth != width) {
            rgb = horizontal(rgb, width, height, Filter.of(width, targetWidth));
            width = targetWidth;
        }
        if (targetHeight != height) rgb = vertical(rgb, width, Filter.of(height, targetHeight));
        return rgb;
    }

    private static int[] horizontal(int[] in, int width, int height, Filter filter) {
        int out = filter.first.length;
        int[] result = new int[height * out * 3];
        Parallel.forLoop(
                height,
                y -> {
                    for (int x = 0; x < out; x++)
                        for (int c = 0; c < 3; c++) {
                            int sum = 1 << (PRECISION - 1);
                            for (int k = 0; k < filter.count[x]; k++)
                                sum +=
                                        in[(y * width + filter.first[x] + k) * 3 + c]
                                                * filter.weights[x * filter.ksize + k];
                            result[(y * out + x) * 3 + c] = clip8(sum);
                        }
                });
        return result;
    }

    private static int[] vertical(int[] in, int width, Filter filter) {
        int out = filter.first.length, row = width * 3;
        int[] result = new int[out * row];
        Parallel.forLoop(
                out,
                y -> {
                    for (int i = 0; i < row; i++) {
                        int sum = 1 << (PRECISION - 1);
                        for (int k = 0; k < filter.count[y]; k++)
                            sum +=
                                    in[(filter.first[y] + k) * row + i]
                                            * filter.weights[y * filter.ksize + k];
                        result[y * row + i] = clip8(sum);
                    }
                });
        return result;
    }

    /** Pillow's fixed point: 32 bits less 8 for the pixel and 2 of accumulation headroom. */
    private static final int PRECISION = 22;

    private static int clip8(int fixedPoint) {
        return Math.max(0, Math.min(255, fixedPoint >> PRECISION));
    }

    /**
     * One axis of the bicubic ({@code a = -0.5}) resampling: per output its first input, its input
     * count and the weights. torchvision rounds the normalized weights to int16 with as many
     * fractional bits as the largest weight allows; scaled to {@link #PRECISION}, the passes above
     * then produce torchvision's bytes.
     */
    private record Filter(int[] first, int[] count, int[] weights, int ksize) {
        static Filter of(int inSize, int outSize) {
            double scale = (double) inSize / outSize, filterScale = Math.max(1.0, scale);
            double support = 2.0 * filterScale;
            int ksize = (int) Math.ceil(support) * 2 + 1;
            int[] first = new int[outSize], count = new int[outSize];
            double[] w = new double[outSize * ksize];
            double max = 0;
            for (int o = 0; o < outSize; o++) {
                double center = (o + 0.5) * scale, sum = 0;
                first[o] = Math.max(0, (int) (center - support + 0.5));
                count[o] = Math.min(inSize, (int) (center + support + 0.5)) - first[o];
                for (int k = 0; k < count[o]; k++)
                    sum += w[o * ksize + k] = cubic((k + first[o] - center + 0.5) / filterScale);
                for (int k = 0; k < count[o]; k++) {
                    if (sum != 0) w[o * ksize + k] /= sum;
                    max = Math.max(max, w[o * ksize + k]);
                }
            }
            int precision = 0;
            while (precision < PRECISION && (int) (0.5 + max * (1 << (precision + 1))) < (1 << 15))
                precision++;
            double fixedPoint = Math.scalb(1.0, precision);
            int[] weights = new int[w.length];
            for (int i = 0; i < w.length; i++)
                weights[i] =
                        (int) (w[i] * fixedPoint + (w[i] < 0 ? -0.5 : 0.5))
                                * (1 << (PRECISION - precision));
            return new Filter(first, count, weights, ksize);
        }
    }

    private static double cubic(double x) {
        double a = -0.5;
        x = Math.abs(x);
        if (x < 1) return ((a + 2) * x - (a + 3)) * x * x + 1;
        if (x < 2) return (((x - 5) * x + 8) * x - 4) * a;
        return 0;
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
