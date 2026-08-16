package com.qxotic.jinfer.models.lfm2;

import com.qxotic.jinfer.Parallel;
import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.boundary.Media;
import com.qxotic.jota.memory.MemoryArena;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/** LiquidAI's aligned resize and optional 512px tile plan. */
final class Lfm2VisionPreprocess {
    static final int MIN_IMAGE_TOKENS = 64;
    static final int MAX_IMAGE_TOKENS = 256;
    static final int TILE_SIZE = 512;
    static final int MAX_TILES = 10;
    static final float MAX_PIXELS_TOLERANCE = 2f;

    private Lfm2VisionPreprocess() {}

    record Part(Media.Image image, int row, int column, boolean thumbnail) {}

    record Plan(List<Part> parts, int rows, int columns) {
        Plan {
            parts = List.copyOf(parts);
        }

        boolean tiled() {
            return rows > 0 && columns > 0;
        }
    }

    static Plan plan(Media.Image image, int patchSize, int merge) {
        int factor = Math.multiplyExact(patchSize, merge);
        int factorArea = Math.multiplyExact(factor, factor);
        int[] overviewSize =
                smartResize(
                        image.width(),
                        image.height(),
                        factor,
                        Math.multiplyExact(MIN_IMAGE_TOKENS, factorArea),
                        Math.multiplyExact(MAX_IMAGE_TOKENS, factorArea));
        Part overview =
                new Part(
                        resizeRegion(
                                image,
                                overviewSize[0],
                                overviewSize[1],
                                0,
                                0,
                                overviewSize[0],
                                overviewSize[1]),
                        0,
                        0,
                        true);
        if (image.width() <= TILE_SIZE * MAX_PIXELS_TOLERANCE
                && image.height() <= TILE_SIZE * MAX_PIXELS_TOLERANCE) {
            return new Plan(List.of(overview), 0, 0);
        }

        int[] grid = closestGrid(image.width(), image.height());
        int columns = grid[0], rows = grid[1];
        int refinedWidth = Math.multiplyExact(TILE_SIZE, columns);
        int refinedHeight = Math.multiplyExact(TILE_SIZE, rows);
        List<Part> parts = new ArrayList<>(rows * columns + 1);
        for (int row = 0; row < rows; row++)
            for (int column = 0; column < columns; column++)
                parts.add(
                        new Part(
                                resizeRegion(
                                        image,
                                        refinedWidth,
                                        refinedHeight,
                                        column * TILE_SIZE,
                                        row * TILE_SIZE,
                                        TILE_SIZE,
                                        TILE_SIZE),
                                row + 1,
                                column + 1,
                                false));
        parts.add(overview);
        return new Plan(parts, rows, columns);
    }

    static int positions(Part part, int patchSize, int merge) {
        int factor = Math.multiplyExact(patchSize, merge);
        return Math.multiplyExact(part.image().width() / factor, part.image().height() / factor);
    }

    static int[] smartResize(int width, int height, int factor, int minPixels, int maxPixels) {
        if (width <= 0 || height <= 0 || factor <= 0 || minPixels <= 0 || maxPixels < minPixels)
            throw new IllegalArgumentException("invalid smart-resize geometry");
        int targetWidth = Math.max(factor, Math.round((float) width / factor) * factor);
        int targetHeight = Math.max(factor, Math.round((float) height / factor) * factor);
        long area = (long) targetWidth * targetHeight;
        if (area > maxPixels) {
            double beta = Math.sqrt((double) width * height / maxPixels);
            targetWidth = Math.max(factor, (int) Math.floor(width / beta / factor) * factor);
            targetHeight = Math.max(factor, (int) Math.floor(height / beta / factor) * factor);
        } else if (area < minPixels) {
            double beta = Math.sqrt((double) minPixels / ((double) width * height));
            targetWidth = (int) Math.ceil(width * beta / factor) * factor;
            targetHeight = (int) Math.ceil(height * beta / factor) * factor;
        }
        return new int[] {targetWidth, targetHeight};
    }

    static int[] closestGrid(int width, int height) {
        float aspect = (float) width / height;
        float bestDifference = Float.MAX_VALUE;
        int bestWidth = 1, bestHeight = 1;
        long area = (long) width * height;
        List<int[]> candidates = new ArrayList<>();
        for (int limit = 1; limit <= MAX_TILES; limit++) {
            for (int widthAtLimit = 1; widthAtLimit < limit; widthAtLimit++)
                if (widthAtLimit * limit <= MAX_TILES)
                    candidates.add(new int[] {widthAtLimit, limit});
            for (int heightAtLimit = 1; heightAtLimit <= limit; heightAtLimit++)
                if (limit * heightAtLimit <= MAX_TILES)
                    candidates.add(new int[] {limit, heightAtLimit});
        }
        candidates.sort(Comparator.comparingInt(value -> value[0] * value[1]));
        for (int[] candidate : candidates) {
            float difference = Math.abs(aspect - (float) candidate[0] / candidate[1]);
            if (difference < bestDifference
                    || (difference == bestDifference
                            && area
                                    > (long) TILE_SIZE
                                            * TILE_SIZE
                                            * candidate[0]
                                            * candidate[1]
                                            / 2)) {
                bestDifference = difference;
                bestWidth = candidate[0];
                bestHeight = candidate[1];
            }
        }
        return new int[] {bestWidth, bestHeight};
    }

    private static Media.Image resizeRegion(
            Media.Image source,
            int targetWidth,
            int targetHeight,
            int regionX,
            int regionY,
            int regionWidth,
            int regionHeight) {
        int sourceWidth = source.width(), sourceHeight = source.height();
        int channels = source.channels();
        float[] input = source.values();
        float[] output =
                new float[Math.multiplyExact(Math.multiplyExact(regionWidth, regionHeight), 3)];
        Parallel.forRows(
                regionHeight,
                y -> {
                    int targetY = regionY + y;
                    float sourceY = (targetY + 0.5f) * sourceHeight / targetHeight - 0.5f;
                    int y0 = Math.max(0, Math.min(sourceHeight - 1, (int) Math.floor(sourceY)));
                    int y1 = Math.min(sourceHeight - 1, y0 + 1);
                    float wy = Math.max(0f, sourceY - y0);
                    for (int x = 0; x < regionWidth; x++) {
                        int targetX = regionX + x;
                        float sourceX = (targetX + 0.5f) * sourceWidth / targetWidth - 0.5f;
                        int x0 = Math.max(0, Math.min(sourceWidth - 1, (int) Math.floor(sourceX)));
                        int x1 = Math.min(sourceWidth - 1, x0 + 1);
                        float wx = Math.max(0f, sourceX - x0);
                        for (int c = 0; c < 3; c++) {
                            int sourceChannel = Math.min(c, channels - 1);
                            float a = input[(y0 * sourceWidth + x0) * channels + sourceChannel];
                            float b = input[(y0 * sourceWidth + x1) * channels + sourceChannel];
                            float d = input[(y1 * sourceWidth + x0) * channels + sourceChannel];
                            float e = input[(y1 * sourceWidth + x1) * channels + sourceChannel];
                            output[(y * regionWidth + x) * 3 + c] =
                                    a * (1 - wx) * (1 - wy)
                                            + b * wx * (1 - wy)
                                            + d * (1 - wx) * wy
                                            + e * wx * wy;
                        }
                    }
                });
        return new Media.Image(output, regionHeight, regionWidth, 3);
    }

    static MemoryView<MemorySegment> patches(
            Media.Image image, int patchSize, MemoryArena<MemorySegment> scratch) {
        int width = image.width(), height = image.height();
        if (patchSize <= 0 || width % patchSize != 0 || height % patchSize != 0)
            throw new IllegalArgumentException("image dimensions must be divisible by patchSize");
        int patchesX = width / patchSize, patchesY = height / patchSize;
        int count = Math.multiplyExact(patchesX, patchesY);
        int patchVector = Math.multiplyExact(3, Math.multiplyExact(patchSize, patchSize));
        float[] pixels = image.values();
        float[] data = new float[Math.multiplyExact(count, patchVector)];
        Parallel.forRows(
                count,
                patch -> {
                    int patchY = patch / patchesX, patchX = patch % patchesX;
                    int at = patch * patchVector;
                    for (int c = 0; c < 3; c++)
                        for (int y = 0; y < patchSize; y++)
                            for (int x = 0; x < patchSize; x++)
                                data[at++] =
                                        pixels[
                                                                ((patchY * patchSize + y) * width
                                                                                        + patchX
                                                                                                * patchSize
                                                                                        + x)
                                                                                * 3
                                                                        + c]
                                                        * 2f
                                                - 1f;
                });
        MemoryView<MemorySegment> result = Views.allocateF32(scratch, count, patchVector);
        Views.copyFromArray(result, 0, data, 0, data.length, "vision patches");
        return result;
    }
}
