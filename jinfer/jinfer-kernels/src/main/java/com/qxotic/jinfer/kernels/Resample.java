package com.qxotic.jinfer.kernels;

import com.qxotic.jinfer.Parallel;
import com.qxotic.jinfer.media.Media;

/**
 * Pillow's separable bicubic resampling ({@code a = -0.5}, antialiased when shrinking) on 8-bit
 * RGB, as llama.cpp's {@code resize_pillow} does it: a horizontal pass then a vertical one, weights
 * in 22-bit fixed point. The reference processors of the vision towers resize so; a tower that
 * resizes otherwise reads other pixels.
 */
public final class Resample {

    private Resample() {}

    /**
     * {@code image} as 8-bit RGB {@code [height][width][3]}, stretched to {@code targetWidth x
     * targetHeight}: alpha dropped and grey replicated, as stb_image loads it. {@code torchWeights}
     * rounds the weights as torchvision does (int16, as many fractional bits as the largest weight
     * allows), else as Pillow does.
     */
    public static int[] bicubic(
            Media.Image image, int targetWidth, int targetHeight, boolean torchWeights) {
        if (targetWidth <= 0 || targetHeight <= 0)
            throw new IllegalArgumentException(
                    "target " + targetWidth + "x" + targetHeight + " is empty");
        int width = image.width(), height = image.height(), channels = image.channels();
        float[] values = image.values();
        int[] rgb = new int[Math.multiplyExact(width * height, 3)];
        for (int p = 0; p < width * height; p++)
            for (int c = 0; c < 3; c++)
                rgb[p * 3 + c] =
                        Math.round(values[p * channels + Math.min(c, channels - 1)] * 255f);
        if (targetWidth != width) {
            rgb = horizontal(rgb, width, height, Filter.of(width, targetWidth, torchWeights));
            width = targetWidth;
        }
        if (targetHeight != height)
            rgb = vertical(rgb, width, Filter.of(height, targetHeight, torchWeights));
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

    /** One axis: per output its first input, its input count and the fixed-point weights. */
    private record Filter(int[] first, int[] count, int[] weights, int ksize) {
        static Filter of(int inSize, int outSize, boolean torchWeights) {
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
            int precision = PRECISION;
            if (torchWeights) { // int16: as many fractional bits as the largest weight allows
                precision = 0;
                while (precision < PRECISION
                        && (int) (0.5 + max * (1 << (precision + 1))) < (1 << 15)) precision++;
            }
            double fixedPoint = Math.scalb(1.0, precision);
            int[] weights = new int[w.length];
            for (int i = 0; i < w.length; i++) // Pillow: +-0.5, then truncate toward zero
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
}
