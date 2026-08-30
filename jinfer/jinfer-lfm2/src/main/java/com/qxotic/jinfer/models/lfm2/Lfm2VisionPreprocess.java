package com.qxotic.jinfer.models.lfm2;

import com.qxotic.jinfer.Parallel;
import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.media.Media;
import com.qxotic.jota.memory.MemoryArena;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;

/** LiquidAI's aligned resize and optional 512px tile plan. */
final class Lfm2VisionPreprocess {
    private static final int MIN_IMAGE_TOKENS = 64;
    private static final int MAX_IMAGE_TOKENS = 256;
    private static final int TILE_SIZE = 512;
    private static final int MAX_TILES = 10;
    private static final float MAX_PIXELS_TOLERANCE = 2f;

    private Lfm2VisionPreprocess() {}

    record Options(
            int minPixels,
            int maxPixels,
            int tileSize,
            int minTiles,
            int maxTiles,
            float maxPixelsTolerance) {
        Options {
            if (minPixels <= 0
                    || maxPixels < minPixels
                    || tileSize <= 0
                    || minTiles <= 0
                    || maxTiles < minTiles
                    || !(maxPixelsTolerance >= 1f)
                    || !Float.isFinite(maxPixelsTolerance))
                throw new IllegalArgumentException("invalid LFM2 vision preprocessing options");
        }
    }

    static Options defaults(int patchSize, int merge) {
        int factor = Math.multiplyExact(patchSize, merge);
        int factorArea = Math.multiplyExact(factor, factor);
        return new Options(
                Math.multiplyExact(MIN_IMAGE_TOKENS, factorArea),
                Math.multiplyExact(MAX_IMAGE_TOKENS, factorArea),
                TILE_SIZE,
                1,
                MAX_TILES,
                MAX_PIXELS_TOLERANCE);
    }

    record Part(Media.Image image, int row, int column) {
        boolean thumbnail() {
            return row == 0;
        }
    }

    record Geometry(int overviewWidth, int overviewHeight, int rows, int columns) {
        boolean tiled() {
            return rows > 0;
        }
    }

    record Plan(List<Part> parts) {
        Plan {
            parts = List.copyOf(parts);
        }

        boolean tiled() {
            return parts.size() > 1;
        }
    }

    static Plan plan(Media.Image image, int patchSize, int merge) {
        return plan(image, patchSize, merge, defaults(patchSize, merge));
    }

    static Plan plan(Media.Image image, int patchSize, int merge, Options options) {
        Geometry geometry = geometry(image, patchSize, merge, options);
        Part overview =
                new Part(
                        resizeRegion(
                                image,
                                geometry.overviewWidth(),
                                geometry.overviewHeight(),
                                0,
                                0,
                                geometry.overviewWidth(),
                                geometry.overviewHeight()),
                        0,
                        0);
        if (!geometry.tiled()) return new Plan(List.of(overview));

        int columns = geometry.columns(), rows = geometry.rows();
        int refinedWidth = Math.multiplyExact(options.tileSize(), columns);
        int refinedHeight = Math.multiplyExact(options.tileSize(), rows);
        List<Part> parts = new ArrayList<>(Math.addExact(Math.multiplyExact(rows, columns), 1));
        for (int row = 0; row < rows; row++)
            for (int column = 0; column < columns; column++)
                parts.add(
                        new Part(
                                resizeRegion(
                                        image,
                                        refinedWidth,
                                        refinedHeight,
                                        column * options.tileSize(),
                                        row * options.tileSize(),
                                        options.tileSize(),
                                        options.tileSize()),
                                row + 1,
                                column + 1));
        parts.add(overview);
        return new Plan(parts);
    }

    static int positions(Media.Image image, int patchSize, int merge, Options options) {
        int factor = Math.multiplyExact(patchSize, merge);
        Geometry geometry = geometry(image, patchSize, merge, options);
        int overview =
                Math.multiplyExact(
                        geometry.overviewWidth() / factor, geometry.overviewHeight() / factor);
        if (!geometry.tiled()) return overview;
        int tileSide = options.tileSize() / factor;
        int tiles = Math.multiplyExact(geometry.rows(), geometry.columns());
        return Math.addExact(
                overview, Math.multiplyExact(tiles, Math.multiplyExact(tileSide, tileSide)));
    }

    private static Geometry geometry(Media.Image image, int patchSize, int merge, Options options) {
        int factor = Math.multiplyExact(patchSize, merge);
        if (options.tileSize() % factor != 0)
            throw new IllegalArgumentException("tile size must be divisible by patchSize * merge");
        int[] overviewSize =
                smartResize(
                        image.width(),
                        image.height(),
                        factor,
                        options.minPixels(),
                        options.maxPixels());
        // the reference processor's _is_image_too_large: the factor-rounded AREA against
        // max_image_tokens * factor^2 * tolerance (an area rule, not a per-side box: an 800x800
        // photo tiles, a 1100x200 banner does not)
        long roundedWidth =
                Math.max(patchSize, Math.round((float) image.width() / factor) * (long) factor);
        long roundedHeight =
                Math.max(patchSize, Math.round((float) image.height() / factor) * (long) factor);
        if (roundedWidth * roundedHeight
                <= options.maxPixels() * (double) options.maxPixelsTolerance()) {
            return new Geometry(overviewSize[0], overviewSize[1], 0, 0);
        }

        int[] grid =
                closestGrid(
                        image.width(),
                        image.height(),
                        options.tileSize(),
                        options.minTiles(),
                        options.maxTiles());
        return new Geometry(overviewSize[0], overviewSize[1], grid[1], grid[0]);
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

    static int[] closestGrid(int width, int height, int tileSize, int minTiles, int maxTiles) {
        float aspect = (float) width / height;
        float bestDifference = Float.MAX_VALUE;
        int bestWidth = 1, bestHeight = 1;
        long area = (long) width * height;
        for (int tiles = minTiles; tiles <= maxTiles; tiles++) {
            for (int candidateWidth = 1; candidateWidth <= tiles; candidateWidth++) {
                if (tiles % candidateWidth != 0) continue;
                int candidateHeight = tiles / candidateWidth;
                float difference = Math.abs(aspect - (float) candidateWidth / candidateHeight);
                if (difference < bestDifference
                        || (difference == bestDifference
                                && area > (long) tileSize * tileSize * tiles / 2)) {
                    bestDifference = difference;
                    bestWidth = candidateWidth;
                    bestHeight = candidateHeight;
                }
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
        Parallel.forLoop(
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
            Media.Image image,
            int patchSize,
            float[] mean,
            float[] std,
            MemoryArena<MemorySegment> scratch) {
        int width = image.width(), height = image.height();
        if (patchSize <= 0 || width % patchSize != 0 || height % patchSize != 0)
            throw new IllegalArgumentException("image dimensions must be divisible by patchSize");
        int patchesX = width / patchSize, patchesY = height / patchSize;
        int count = Math.multiplyExact(patchesX, patchesY);
        int patchVector = Math.multiplyExact(3, Math.multiplyExact(patchSize, patchSize));
        float[] pixels = image.values();
        float[] data = new float[Math.multiplyExact(count, patchVector)];
        Parallel.forLoop(
                count,
                patch -> {
                    int patchY = patch / patchesX, patchX = patch % patchesX;
                    int at = patch * patchVector;
                    for (int c = 0; c < 3; c++)
                        for (int y = 0; y < patchSize; y++)
                            for (int x = 0; x < patchSize; x++)
                                data[at++] =
                                        (pixels[
                                                                ((patchY * patchSize + y) * width
                                                                                        + patchX
                                                                                                * patchSize
                                                                                        + x)
                                                                                * 3
                                                                        + c]
                                                        - mean[c])
                                                / std[c];
                });
        MemoryView<MemorySegment> result = Views.allocateF32(scratch, count, patchVector);
        Views.copyFromArray(result, 0, data, 0, data.length, "vision patches");
        return result;
    }
}
