package com.qxotic.jinfer.models.gemma4;

import com.qxotic.jinfer.Parallel;
import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.media.Media;
import com.qxotic.jota.memory.MemoryArena;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;

/**
 * Shared Gemma 4 image preprocessing as Google's reference does it (the JAX {@code
 * gemma4/vision/_preprocessing.py}, HF's {@code Gemma4ImageProcessor}), and patch extraction.
 */
final class VisionPreprocess {
    private VisionPreprocess() {}

    static final int IMAGE_TOKEN_BUDGET = validatedBudget("jinfer.gemma4.imageTokenBudget", 280);
    static final int VIDEO_TOKEN_BUDGET = validatedBudget("jinfer.gemma4.videoTokenBudget", -1);

    static int validatedBudget(String property, int defaultValue) {
        String value = System.getProperty(property);
        if (value == null) return defaultValue;
        int budget;
        try {
            budget = Integer.parseInt(value.trim());
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(
                    property + " must be 70|140|280|560|1120, got '" + value + "'");
        }
        if (budget != 70 && budget != 140 && budget != 280 && budget != 560 && budget != 1120)
            throw new IllegalArgumentException(
                    property + " must be 70|140|280|560|1120, got " + budget);
        return budget;
    }

    /** A video frame's budget: the property when set, else the model's video processor default. */
    static int videoBudget(int defaultValue) {
        return VIDEO_TOKEN_BUDGET > 0 ? VIDEO_TOKEN_BUDGET : defaultValue;
    }

    /**
     * The reference's {@code get_aspect_ratio_preserving_size}: the largest {@code [width, height]}
     * with the image's aspect, both edges multiples of {@code side} (one output token), and at most
     * {@code maxTokens} tokens - small images grow to fill the budget, large ones shrink to it. A
     * sliver too thin for one token keeps one along its short edge.
     */
    static int[] targetSize(int width, int height, int side, int maxTokens) {
        if (width <= 0 || height <= 0 || side <= 0 || maxTokens <= 0)
            throw new IllegalArgumentException("invalid image geometry");
        double factor = Math.sqrt((double) maxTokens * side * side / ((double) height * width));
        int targetHeight = (int) Math.floor(factor * height / side) * side;
        int targetWidth = (int) Math.floor(factor * width / side) * side;
        int maxSide = maxTokens * side;
        if (targetHeight == 0) {
            targetHeight = side;
            targetWidth = Math.min(Math.max(1, width / height) * side, maxSide);
        } else if (targetWidth == 0) {
            targetWidth = side;
            targetHeight = Math.min(Math.max(1, height / width) * side, maxSide);
        }
        return new int[] {targetWidth, targetHeight};
    }

    /** Source rows per horizontal-pass block: each tap is then one loop over 3 * ROWS values. */
    private static final int ROWS = 16;

    /**
     * The image as {@code [3][targetHeight][targetWidth]} planes in {@code [0,1]}: 8-bit RGB
     * resized to the target with Pillow's bicubic (fixed point, horizontal then vertical), as the
     * reference does it.
     */
    static float[] toCHW(Media.Image image, int targetWidth, int targetHeight) {
        if (targetWidth <= 0 || targetHeight <= 0)
            throw new IllegalArgumentException("target dimensions must be positive");
        int width = image.width(), height = image.height(), channels = image.channels();
        float[] values = image.values();
        int w = targetWidth, h = targetHeight, plane = w * h, lane = 3 * ROWS;
        // horizontal pass, ROWS source rows at a time, rounded to 8 bits into a column-major
        // block [x][row * 3 + c] (never the whole image), into [height][w][3]
        Filter across = Filter.of(width, w);
        int[] rows = new int[Math.multiplyExact(height, 3 * w)];
        int[][] blocks = new int[Parallel.threads()][lane * width];
        int[][] sums = new int[Parallel.threads()][Math.max(lane, 3 * w)];
        Parallel.forLoop(
                (height + ROWS - 1) / ROWS,
                (block, slot) -> {
                    int y0 = block * ROWS, n = Math.min(ROWS, height - y0);
                    int[] columns = blocks[slot], sum = sums[slot];
                    for (int r = 0; r < n; r++)
                        for (int x = 0; x < width; x++)
                            for (int c = 0; c < 3; c++)
                                columns[x * lane + r * 3 + c] =
                                        Math.round(
                                                values[
                                                                ((y0 + r) * width + x) * channels
                                                                        + Math.min(c, channels - 1)]
                                                        * 255f);
                    for (int x = 0; x < w; x++) {
                        across.apply(columns, lane, x, sum);
                        for (int r = 0; r < n; r++)
                            for (int c = 0; c < 3; c++)
                                rows[((y0 + r) * w + x) * 3 + c] = sum[r * 3 + c];
                    }
                });
        // vertical pass: a whole output row per tap, then into the [0,1] planes
        Filter down = Filter.of(height, h);
        float[] out = new float[3 * plane];
        Parallel.forLoop(
                h,
                (y, slot) -> {
                    int[] sum = sums[slot];
                    down.apply(rows, 3 * w, y, sum);
                    for (int x = 0; x < w; x++)
                        for (int c = 0; c < 3; c++)
                            out[c * plane + y * w + x] = sum[x * 3 + c] / 255.0f;
                });
        return out;
    }

    private static final int PRECISION_BITS = 22; // Pillow's fixed point: 32 - 8 - 2

    /**
     * One axis of Pillow's separable bicubic ({@code a = -0.5}) on 8-bit values: per output, its
     * first input and fixed-point weights ({@code precompute_coeffs}). An unchanged size copies
     * exactly: the kernel is 1 at distance 0 and 0 at 1 and 2, so the one weight is 2^22.
     */
    private record Filter(int[] first, int[] count, int[] weights, int ksize) {
        static Filter of(int inSize, int outSize) {
            double scale = (double) inSize / outSize, filterScale = Math.max(1.0, scale);
            double support = 2.0 * filterScale, inverse = 1.0 / filterScale;
            int ksize = (int) Math.ceil(support) * 2 + 1;
            int[] first = new int[outSize],
                    count = new int[outSize],
                    weights = new int[outSize * ksize];
            double[] w = new double[ksize];
            for (int o = 0; o < outSize; o++) {
                double center = (o + 0.5) * scale, sum = 0;
                first[o] = Math.max(0, (int) (center - support + 0.5));
                count[o] = Math.min(inSize, (int) (center + support + 0.5)) - first[o];
                for (int k = 0; k < count[o]; k++)
                    sum += w[k] = cubic((k + first[o] - center + 0.5) * inverse);
                for (int k = 0; k < count[o]; k++) {
                    double v = sum != 0 ? w[k] / sum : w[k];
                    weights[o * ksize + k] =
                            (int) (v * (1 << PRECISION_BITS) + (v < 0 ? -0.5 : 0.5));
                }
            }
            return new Filter(first, count, weights, ksize);
        }

        /**
         * Output {@code o} of {@code lane} interleaved lines at once, into {@code sum[0, lane)}:
         * input {@code i} of every line sits at {@code in[i * lane, (i + 1) * lane)}, so each tap
         * is one loop over the lines (what the JIT vectorizes).
         */
        void apply(int[] in, int lane, int o, int[] sum) {
            Arrays.fill(sum, 0, lane, 1 << (PRECISION_BITS - 1));
            for (int k = 0; k < count[o]; k++) {
                int weight = weights[o * ksize + k], base = (first[o] + k) * lane;
                for (int j = 0; j < lane; j++) sum[j] += in[base + j] * weight;
            }
            for (int j = 0; j < lane; j++) sum[j] = clip8(sum[j]);
        }
    }

    private static double cubic(double x) {
        double a = -0.5;
        x = Math.abs(x);
        if (x < 1) return ((a + 2) * x - (a + 3)) * x * x + 1;
        if (x < 2) return (((x - 5) * x + 8) * x - 4) * a;
        return 0;
    }

    private static int clip8(int fixedPoint) {
        return Math.max(0, Math.min(255, fixedPoint >> PRECISION_BITS));
    }

    static MemoryView<MemorySegment> im2col(
            Media.Image image,
            int targetWidth,
            int targetHeight,
            int patchSize,
            MemoryArena<MemorySegment> scratch) {
        if (patchSize <= 0 || targetWidth % patchSize != 0 || targetHeight % patchSize != 0)
            throw new IllegalArgumentException("target dimensions must be divisible by patchSize");
        int patchesX = targetWidth / patchSize, patchesY = targetHeight / patchSize;
        int count = Math.multiplyExact(patchesX, patchesY);
        int patchVector = Math.multiplyExact(3, Math.multiplyExact(patchSize, patchSize));
        int plane = Math.multiplyExact(targetHeight, targetWidth);
        float[] chw = toCHW(image, targetWidth, targetHeight);
        // Gather into a heap buffer first (parallel-safe), then ONE bulk copy on the owning
        // thread: a checked copy inside forLoop would trip thread-confinement on confined arenas.
        float[] data = new float[Math.multiplyExact(count, patchVector)];
        Parallel.forLoop(
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
