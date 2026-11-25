package com.qxotic.jinfer.models.inkling2;

import static com.qxotic.jinfer.models.inkling2.Inkling2.f32;
import static com.qxotic.jinfer.models.inkling2.Inkling2.require;
import static com.qxotic.jinfer.models.inkling2.Inkling2.weight;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.Arenas;
import com.qxotic.jinfer.Parallel;
import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.kernels.Convert;
import com.qxotic.jinfer.kernels.FlashAttention;
import com.qxotic.jinfer.kernels.GeluErf;
import com.qxotic.jinfer.kernels.MatMul;
import com.qxotic.jinfer.kernels.ModelLoader;
import com.qxotic.jinfer.kernels.Norms;
import com.qxotic.jinfer.kernels.Ops;
import com.qxotic.jinfer.kernels.Resample;
import com.qxotic.jinfer.kernels.RoPE;
import com.qxotic.jinfer.kernels.Trace;
import com.qxotic.jinfer.media.Media;
import com.qxotic.jinfer.media.MediaProjector;
import com.qxotic.jota.memory.MemoryArena;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;
import java.lang.ref.Reference;
import java.util.Map;
import java.util.function.Consumer;

/**
 * The Inkling 2 vision tower ({@code clip.vision.projector_type = inkling2v}), after llama.cpp's
 * {@code inkling2v.cpp}: a ViT over square patches with a class token, a learned position table
 * resized to the patch grid, 2D RoPE on Q and K and a LayerScale on both branches. The class token
 * is then dropped, the patches merged in squares and projected by a GELU MLP to the language
 * model's width.
 *
 * <p>An image keeps its size: it is padded right and bottom to whole merged patches, and resized
 * only when that grid passes {@link #MAX_TOKENS}.
 */
final class Inkling2Vision implements MediaProjector<Media.Image> {
    /** llama.cpp's cap on the rows of one image. */
    private static final int MAX_TOKENS = 4096;

    private static final float ROPE_THETA = 10_000f;

    private final int patch, merge, dim, heads, headDim, ffnDim, side, hidden, modelDim;
    private final float eps;
    private final float[] mean, std;
    private final MemoryView<MemorySegment> patchKernel, classEmbedding, classPosition, positions;
    private final MemoryView<MemorySegment> preNorm, preNormBias, projectorUp, projectorDown;
    private final Layer[] layers;

    record Linear(MemoryView<MemorySegment> weight, MemoryView<MemorySegment> bias) {}

    record Layer(
            MemoryView<MemorySegment> norm1,
            MemoryView<MemorySegment> norm1Bias,
            Linear query,
            Linear key,
            Linear value,
            Linear output,
            MemoryView<MemorySegment> scale1,
            MemoryView<MemorySegment> norm2,
            MemoryView<MemorySegment> norm2Bias,
            Linear up,
            Linear down,
            MemoryView<MemorySegment> scale2) {}

    Inkling2Vision(GGUF gguf, Map<String, MemoryView<MemorySegment>> tensors, int modelDim) {
        String v = "clip.vision.";
        require(
                "inkling2v".equals(gguf.getStringOrDefault(v + "projector_type", "")),
                "the mmproj is not an inkling2v vision tower");
        patch = gguf.getValue(int.class, v + "patch_size");
        merge = gguf.getValue(int.class, v + "spatial_merge_size");
        dim = gguf.getValue(int.class, v + "embedding_length");
        heads = gguf.getValue(int.class, v + "attention.head_count");
        ffnDim = gguf.getValue(int.class, v + "feed_forward_length");
        eps = gguf.getValue(float.class, v + "attention.layer_norm_epsilon");
        mean = gguf.getValue(float[].class, v + "image_mean");
        std = gguf.getValue(float[].class, v + "image_std");
        this.modelDim = modelDim;
        headDim = dim / heads;
        require(
                patch > 0 && merge > 0 && dim % heads == 0 && headDim % 4 == 0,
                "invalid vision geometry");
        require(
                gguf.getValue(int.class, v + "projection_dim") == modelDim,
                "vision projection width != model width " + modelDim);
        require(
                !tensors.containsKey("mm.1.bias") && !tensors.containsKey("mm.2.bias"),
                "projector biases are not supported");

        int patchVector = 3 * patch * patch;
        patchKernel = f32(tensors, "v.patch_embd.weight", (long) dim * patchVector);
        classEmbedding = f32(tensors, "v.class_embd", dim);
        classPosition = f32(tensors, "v.class_position_embd.weight", dim);
        positions = ModelLoader.requireF32(tensors, "v.position_embd.weight");
        side = (int) Math.sqrt(positions.shape().size() / dim);
        require(
                positions.shape().size() == (long) side * side * dim,
                "the position table is not a square grid");
        preNorm = f32(tensors, "v.pre_ln.weight", dim);
        preNormBias = f32(tensors, "v.pre_ln.bias", dim);
        layers = new Layer[gguf.getValue(int.class, v + "block_count")];
        for (int l = 0; l < layers.length; l++) {
            String p = "v.blk." + l + ".";
            layers[l] =
                    new Layer(
                            f32(tensors, p + "ln1.weight", dim),
                            f32(tensors, p + "ln1.bias", dim),
                            linear(tensors, p + "attn_q", dim, dim),
                            linear(tensors, p + "attn_k", dim, dim),
                            linear(tensors, p + "attn_v", dim, dim),
                            linear(tensors, p + "attn_out", dim, dim),
                            f32(tensors, p + "ls1.weight", dim),
                            f32(tensors, p + "ln2.weight", dim),
                            f32(tensors, p + "ln2.bias", dim),
                            linear(tensors, p + "ffn_up", ffnDim, dim),
                            linear(tensors, p + "ffn_down", dim, ffnDim),
                            f32(tensors, p + "ls2.weight", dim));
        }
        MemoryView<MemorySegment> up = ModelLoader.require(tensors, "mm.1.weight");
        hidden = Math.toIntExact(up.dataType().logicalShape(up.shape()).flatAt(0));
        projectorUp = weight(tensors, "mm.1.weight", hidden, dim * merge * merge);
        projectorDown = weight(tensors, "mm.2.weight", modelDim, hidden);
    }

    private static Linear linear(
            Map<String, MemoryView<MemorySegment>> tensors, String name, int rows, int columns) {
        return new Linear(
                weight(tensors, name + ".weight", rows, columns),
                f32(tensors, name + ".bias", rows));
    }

    /**
     * {@code {width, height}} the image is drawn at: its own, or within the token cap as
     * llama.cpp's {@code calc_size_preserved_ratio} sizes it, in single precision.
     */
    int[] size(Media.Image image) {
        int cell = patch * merge, width = image.width(), height = image.height();
        int maxPixels = MAX_TOKENS * cell * cell;
        if ((long) padded(width) * padded(height) <= maxPixels) return new int[] {width, height};
        int w = Math.max(cell, Math.round((float) width / cell) * cell);
        int h = Math.max(cell, Math.round((float) height / cell) * cell);
        if ((long) w * h > maxPixels) {
            float beta = (float) Math.sqrt((float) height * width / maxPixels);
            w = Math.max(cell, (int) Math.floor(width / beta / cell) * cell);
            h = Math.max(cell, (int) Math.floor(height / beta / cell) * cell);
        }
        return new int[] {w, h};
    }

    /** {@code pixels} rounded up to whole merged patches. */
    private int padded(int pixels) {
        int cell = patch * merge;
        return (pixels + cell - 1) / cell * cell;
    }

    @Override
    public int positions(Media.Image image) {
        int[] size = size(image);
        int cell = patch * merge;
        return padded(size[0]) / cell * (padded(size[1]) / cell);
    }

    @Override
    public String planId() {
        return "inkling2v patch=" + patch + " merge=" + merge + " tokens<=" + MAX_TOKENS;
    }

    @Override
    public void project(Media.Image image, int maxChunkSize, Consumer<MemoryView<?>> sink) {
        if (maxChunkSize <= 0) throw new IllegalArgumentException("maxChunkSize must be positive");
        int[] size = size(image);
        int width = padded(size[0]) / patch, height = padded(size[1]) / patch;
        int rows = width / merge * (height / merge);
        MemoryArena<MemorySegment> scratch = Arenas.newCrossThreadMemoryArena();
        try {
            MemoryView<MemorySegment> out =
                    encode(patches(image, size, width, height), width, height, scratch);
            for (int first = 0; first < rows; first += maxChunkSize)
                sink.accept(out.slice(0, first, Math.min(rows, first + maxChunkSize)));
        } finally {
            Arenas.close(scratch);
            Reference.reachabilityFence(this);
        }
    }

    /**
     * The normalized patch vectors of a {@code width x height} patch grid, raster order, channel
     * then row then column within a patch (the kernel's layout). The image is drawn at {@code size}
     * in the top left; the rest is the reference's padding, a pixel of -1 before normalization.
     */
    private float[] patches(Media.Image image, int[] size, int width, int height) {
        int drawnWidth = size[0], drawnHeight = size[1], area = patch * patch;
        int[] rgb = Resample.bicubic(image, drawnWidth, drawnHeight, false);
        float[] out = new float[width * height * 3 * area];
        Parallel.forLoop(
                height,
                py -> {
                    for (int px = 0; px < width; px++) {
                        int at = (py * width + px) * 3 * area;
                        for (int c = 0; c < 3; c++)
                            for (int y = py * patch; y < (py + 1) * patch; y++)
                                for (int x = px * patch; x < (px + 1) * patch; x++) {
                                    float value =
                                            x < drawnWidth && y < drawnHeight
                                                    ? rgb[(y * drawnWidth + x) * 3 + c]
                                                    : -1f;
                                    out[at++] = (value - mean[c] * 255f) / (std[c] * 255f);
                                }
                    }
                });
        return out;
    }

    /**
     * The projected rows of a {@code width x height} grid of patch vectors, allocated in {@code
     * arena}.
     */
    MemoryView<MemorySegment> encode(
            float[] patches, int width, int height, MemoryArena<MemorySegment> arena) {
        int count = width * height, n = count + 1, patchVector = 3 * patch * patch;
        MemoryView<MemorySegment> pixels = Views.allocateF32(arena, count, patchVector);
        Views.copyFromArray(pixels, 0, patches, 0, patches.length, "patches");
        MemoryView<MemorySegment> x = Views.allocateF32(arena, n, dim);
        MemoryView<MemorySegment> normed = Views.allocateF32(arena, n, dim);
        MemoryView<MemorySegment> branch = Views.allocateF32(arena, n, dim);
        MemoryView<MemorySegment> query = Views.allocateF32(arena, n, dim);
        MemoryView<MemorySegment> key = Views.allocateF32(arena, n, dim);
        MemoryView<MemorySegment> value = Views.allocateF32(arena, n, dim);
        MemoryView<MemorySegment> wide = Views.allocateF32(arena, n, ffnDim);

        // the class token, then the patches with their positions
        Convert.copyF32(classEmbedding, 0, x, 0, dim);
        Ops.addInPlace(x, 0, classPosition, 0, dim);
        MatMul.mm(
                patchKernel,
                0,
                patchVector,
                pixels,
                0,
                patchVector,
                x,
                dim,
                dim,
                dim,
                count,
                patchVector);
        addPositions(x, width, height);
        Norms.layerNormRows(x, x, preNorm, preNormBias, n, dim, eps);
        if (Trace.ENABLED) Trace.sum("pre_ln", x, n * dim);

        // RoPE positions: the class token is at (0, 0) and the patches count from 1
        int lanes = headDim / 4, grid = Math.max(width, height) + 1;
        MemoryView<MemorySegment> cos = Views.allocateF32(arena, grid, lanes);
        MemoryView<MemorySegment> sin = Views.allocateF32(arena, grid, lanes);
        RoPE.fill(cos, sin, 0, grid, lanes, RoPE.plain(headDim / 2, ROPE_THETA));
        float scale = 1f / (float) Math.sqrt(headDim);

        for (int l = 0; l < layers.length; l++) {
            Layer layer = layers[l];
            Norms.layerNormRows(normed, x, layer.norm1, layer.norm1Bias, n, dim, eps);
            apply(layer.query, normed, query, n, dim);
            apply(layer.key, normed, key, n, dim);
            apply(layer.value, normed, value, n, dim);
            Parallel.forLoop(
                    n,
                    token -> {
                        int column = token == 0 ? 0 : (token - 1) % width + 1;
                        int row = token == 0 ? 0 : (token - 1) / width + 1;
                        for (int head = 0; head < heads; head++) {
                            long at = (long) token * dim + (long) head * headDim;
                            rotate(query, at, column, row, cos, sin);
                            rotate(key, at, column, row, cos, sin);
                        }
                    });
            FlashAttention.bidirectionalPrefill(
                    query, normed, key, value, heads, n, headDim, dim, dim, 1, scale);
            apply(layer.output, normed, branch, n, dim);
            addScaled(x, branch, layer.scale1, n);

            Norms.layerNormRows(normed, x, layer.norm2, layer.norm2Bias, n, dim, eps);
            apply(layer.up, normed, wide, n, ffnDim);
            GeluErf.rows(wide, n, ffnDim);
            apply(layer.down, wide, branch, n, dim);
            addScaled(x, branch, layer.scale2, n);
            if (Trace.ENABLED) Trace.sum("layer_out-" + l, x, n * dim);
        }

        // squares of the raster grid into one row each: (y, x), (y, x+1), (y+1, x), (y+1, x+1)
        int outX = width / merge, outY = height / merge, tokens = outX * outY;
        MemoryView<MemorySegment> merged = Views.allocateF32(arena, tokens, dim * merge * merge);
        Parallel.forLoop(
                outY,
                oy -> {
                    for (int ox = 0; ox < outX; ox++)
                        for (int iy = 0; iy < merge; iy++)
                            for (int ix = 0; ix < merge; ix++)
                                Convert.copyF32(
                                        x,
                                        (1 + (long) (oy * merge + iy) * width + ox * merge + ix)
                                                * dim,
                                        merged,
                                        (((long) oy * outX + ox) * merge * merge + iy * merge + ix)
                                                * dim,
                                        dim);
                });
        MemoryView<MemorySegment> projected = Views.allocateF32(arena, tokens, hidden);
        MemoryView<MemorySegment> output = Views.allocateF32(arena, tokens, modelDim);
        MatMul.gemm(projectorUp, merged, projected, tokens);
        GeluErf.rows(projected, tokens, hidden);
        MatMul.gemm(projectorDown, projected, output, tokens);
        if (Trace.ENABLED) Trace.sum("projected", output, tokens * modelDim);
        return output;
    }

    /** 2D RoPE of one head: its first half turns with the column, its second with the row. */
    private void rotate(
            MemoryView<MemorySegment> t,
            long at,
            int column,
            int row,
            MemoryView<MemorySegment> cos,
            MemoryView<MemorySegment> sin) {
        RoPE.applyInterleaved(t, at, column, cos, sin, headDim / 4);
        RoPE.applyInterleaved(t, at + headDim / 2, row, cos, sin, headDim / 4);
    }

    /**
     * Adds the position table to the patches of {@code x}, resized to their grid by ggml's
     * half-pixel bilinear interpolation.
     */
    private void addPositions(MemoryView<MemorySegment> x, int width, int height) {
        Parallel.forLoop(
                height,
                py -> {
                    float sy = (py + .5f) * side / height - .5f;
                    int y0 = clamp((int) Math.floor(sy)), y1 = clamp((int) Math.floor(sy) + 1);
                    float dy = Math.clamp(sy - y0, 0f, 1f);
                    for (int px = 0; px < width; px++) {
                        float sx = (px + .5f) * side / width - .5f;
                        int x0 = clamp((int) Math.floor(sx)), x1 = clamp((int) Math.floor(sx) + 1);
                        float dx = Math.clamp(sx - x0, 0f, 1f);
                        long at = (1 + (long) py * width + px) * dim;
                        add(x, at, y0, x0, (1 - dx) * (1 - dy));
                        add(x, at, y0, x1, dx * (1 - dy));
                        add(x, at, y1, x0, (1 - dx) * dy);
                        add(x, at, y1, x1, dx * dy);
                    }
                });
    }

    private int clamp(int cell) {
        return Math.clamp(cell, 0, side - 1);
    }

    private void add(MemoryView<MemorySegment> x, long at, int row, int column, float weight) {
        Ops.saxpyInPlace(x, at, positions, ((long) row * side + column) * dim, dim, weight);
    }

    private static void apply(
            Linear linear,
            MemoryView<MemorySegment> in,
            MemoryView<MemorySegment> out,
            int rows,
            int width) {
        MatMul.gemm(linear.weight, in, out, rows);
        Ops.addRowBiasInPlace(out, 0, linear.bias, 0, rows, width);
    }

    /** {@code x += scale * branch} per row; {@code branch} is consumed. */
    private void addScaled(
            MemoryView<MemorySegment> x,
            MemoryView<MemorySegment> branch,
            MemoryView<MemorySegment> scale,
            int rows) {
        Parallel.forLoop(
                rows,
                row -> {
                    long at = (long) row * dim;
                    Ops.multiplyInPlace(branch, at, scale, 0, dim);
                    Ops.addInPlace(x, at, branch, at, dim);
                });
    }
}
