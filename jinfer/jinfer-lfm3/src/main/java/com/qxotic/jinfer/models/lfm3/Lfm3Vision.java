package com.qxotic.jinfer.models.lfm3;

import static com.qxotic.jinfer.models.lfm3.Lfm3.f32;
import static com.qxotic.jinfer.models.lfm3.Lfm3.require;
import static com.qxotic.jinfer.models.lfm3.Lfm3.weight;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.Arenas;
import com.qxotic.jinfer.Parallel;
import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.kernels.Convert;
import com.qxotic.jinfer.kernels.FlashAttention;
import com.qxotic.jinfer.kernels.MatMul;
import com.qxotic.jinfer.kernels.ModelLoader;
import com.qxotic.jinfer.kernels.Norms;
import com.qxotic.jinfer.kernels.Ops;
import com.qxotic.jinfer.media.Media;
import com.qxotic.jinfer.media.MediaProjector;
import com.qxotic.jota.DataType;
import com.qxotic.jota.memory.MemoryAllocators;
import com.qxotic.jota.memory.MemoryArena;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.ref.Reference;
import java.util.Arrays;
import java.util.Map;
import java.util.function.Consumer;
import java.util.function.ObjIntConsumer;

/**
 * The LFM3-VL vision tower: 16-pixel patches (both temporal kernels of the patchifier applied to
 * the still image) and an RMS embedding norm, then LFM-style blocks - a gated 3x3 depthwise
 * convolution over the patch grid, or GQA with 2D RoPE - each followed by a SiLU-gated FFN, then a
 * LayerNorm, a 2x2 patch merge and a GELU MLP into the language model's width.
 *
 * <pre>
 *   conv block:  B | C | x = in_proj(x);  out_proj(C * dwconv3x3(B * x))
 *   attention:   wo(attend(rope2d(qn(wq x)), rope2d(kn(wk x)), wv x))    rows y, then x
 * </pre>
 */
final class Lfm3Vision implements MediaProjector<Media.Image> {

    private static final int MERGE = 2;

    private final int patch, dim, heads, kvHeads, headDim, ffnDim, projectorHidden, modelDim;
    private final long minPixels, maxPixels;
    private final float eps;
    private final boolean halfPixels;
    private final float[] mean, std, ropeInverseFrequency;
    private final MemoryView<MemorySegment> patchKernel, embeddingNorm;
    private final MemoryView<MemorySegment> projectorNorm, projectorNormBias;
    private final MemoryView<MemorySegment> projectorUp,
            projectorUpBias,
            projectorDown,
            projectorDownBias;
    private final Layer[] layers;

    /** Exactly one of the conv and attention weight groups is set. */
    record Layer(
            MemoryView<MemorySegment> norm1,
            MemoryView<MemorySegment> convIn,
            float[] convTaps, // [9][dim]: tap ky * 3 + kx of every channel, contiguous
            MemoryView<MemorySegment> convOut,
            MemoryView<MemorySegment> query,
            MemoryView<MemorySegment> key,
            MemoryView<MemorySegment> value,
            MemoryView<MemorySegment> output,
            MemoryView<MemorySegment> queryNorm,
            MemoryView<MemorySegment> keyNorm,
            MemoryView<MemorySegment> norm2,
            MemoryView<MemorySegment> ffnGate,
            MemoryView<MemorySegment> ffnUp,
            MemoryView<MemorySegment> ffnDown) {}

    Lfm3Vision(
            GGUF gguf, Map<String, MemoryView<MemorySegment>> tensors, int modelDim, Arena arena) {
        require(
                "lfm3vl".equals(gguf.getString("clip.vision.projector_type")),
                "unsupported vision projector " + gguf.getString("clip.vision.projector_type"));
        patch = gguf.getValue(int.class, "clip.vision.patch_size");
        dim = gguf.getValue(int.class, "clip.vision.embedding_length");
        heads = gguf.getValue(int.class, "clip.vision.attention.head_count");
        kvHeads = gguf.getValue(int.class, "clip.vision.attention.head_count_kv");
        ffnDim = gguf.getValue(int.class, "clip.vision.feed_forward_length");
        eps = gguf.getValue(float.class, "clip.vision.attention.layer_norm_epsilon");
        this.modelDim = modelDim;
        headDim = dim / heads;
        require(
                dim % heads == 0 && heads % kvHeads == 0 && headDim % 4 == 0,
                "vision head geometry");
        require(
                gguf.getValueOrDefault(int.class, "clip.vision.spatial_merge_size", MERGE) == MERGE,
                "the vision merger is 2x2");
        require(
                gguf.getValue(int.class, "clip.vision.projection_dim") == modelDim,
                "vision projection width != model width " + modelDim);
        require(
                "bicubic"
                        .equals(
                                gguf.getStringOrDefault(
                                        "clip.vision.image_resize_algo", "bicubic")),
                "only the bicubic resize is supported");
        mean = gguf.getValue(float[].class, "clip.vision.image_mean");
        std = gguf.getValue(float[].class, "clip.vision.image_std");
        // llama.cpp caps an image at 4096 tokens as an OOM guard
        long patchArea = (long) patch * patch * MERGE * MERGE;
        minPixels = gguf.getValue(int.class, "clip.vision.image_min_pixels");
        maxPixels =
                Math.min(4096, gguf.getValue(int.class, "clip.vision.image_max_pixels") / patchArea)
                        * patchArea;
        require(minPixels > 0 && maxPixels >= minPixels, "invalid image min/max pixels");

        // a still image fills both temporal slices, so the two patchifier kernels add up
        int patchVector = 3 * patch * patch;
        halfPixels =
                ModelLoader.require(tensors, "v.patch_embd.weight").dataType() != DataType.BF16;
        patchKernel = Views.allocateF32(MemoryAllocators.ofArena(arena), dim, patchVector);
        Convert.copyToF32(
                kernel(tensors, "v.patch_embd.weight", patchVector),
                0,
                patchKernel,
                0,
                dim * patchVector);
        if (tensors.containsKey("v.patch_embd.weight.1")) {
            try (Arena temporary = Arena.ofConfined()) {
                MemoryView<MemorySegment> second =
                        Views.allocateF32(MemoryAllocators.ofArena(temporary), dim, patchVector);
                Convert.copyToF32(
                        kernel(tensors, "v.patch_embd.weight.1", patchVector),
                        0,
                        second,
                        0,
                        dim * patchVector);
                Ops.addInPlace(patchKernel, 0, second, 0, dim * patchVector);
            }
        }
        embeddingNorm = f32(tensors, "v.embedding_norm.weight", dim);

        int count = gguf.getValue(int.class, "clip.vision.block_count");
        layers = new Layer[count];
        int kvDim = kvHeads * headDim;
        for (int l = 0; l < count; l++) {
            String p = "v.blk." + l + ".";
            boolean conv = tensors.containsKey(p + "conv2d_in_proj.weight");
            layers[l] =
                    new Layer(
                            f32(tensors, p + "ln1.weight", dim),
                            conv
                                    ? weight(tensors, p + "conv2d_in_proj.weight", 3 * dim, dim)
                                    : null,
                            conv
                                    ? Towers.tapMajor(
                                            Views.toFloatArray(
                                                    f32(
                                                            tensors,
                                                            p + "conv2d_spatial.weight",
                                                            dim,
                                                            1,
                                                            3,
                                                            3),
                                                    "conv2d_spatial"),
                                            dim)
                                    : null,
                            conv ? weight(tensors, p + "conv2d_out_proj.weight", dim, dim) : null,
                            conv ? null : weight(tensors, p + "attn_q.weight", dim, dim),
                            conv ? null : weight(tensors, p + "attn_k.weight", kvDim, dim),
                            conv ? null : weight(tensors, p + "attn_v.weight", kvDim, dim),
                            conv ? null : weight(tensors, p + "attn_out.weight", dim, dim),
                            conv ? null : f32(tensors, p + "attn_q_norm.weight", headDim),
                            conv ? null : f32(tensors, p + "attn_k_norm.weight", headDim),
                            f32(tensors, p + "ln2.weight", dim),
                            weight(tensors, p + "ffn_gate.weight", ffnDim, dim),
                            weight(tensors, p + "ffn_up.weight", ffnDim, dim),
                            weight(tensors, p + "ffn_down.weight", dim, ffnDim));
        }
        projectorHidden =
                Math.toIntExact(ModelLoader.require(tensors, "mm.0.bias").shape().flatAt(0));
        projectorNorm = f32(tensors, "mm.input_norm.weight", dim);
        projectorNormBias = f32(tensors, "mm.input_norm.bias", dim);
        projectorUp = weight(tensors, "mm.0.weight", projectorHidden, dim * MERGE * MERGE);
        projectorUpBias = f32(tensors, "mm.0.bias", projectorHidden);
        projectorDown = weight(tensors, "mm.2.weight", modelDim, projectorHidden);
        projectorDownBias = f32(tensors, "mm.2.bias", modelDim);

        // ggml_rope_multi gets n_dims = headDim / 2, and each section restarts at base^0
        ropeInverseFrequency = new float[headDim / 4];
        for (int j = 0; j < ropeInverseFrequency.length; j++)
            ropeInverseFrequency[j] = (float) Math.pow(1e6, -2.0 * j / (headDim / 2.0));
    }

    /**
     * The tower of an LFM3 mmproj; {@code arena} holds the one derived weight, the patch kernel.
     */
    private int[] targetSize(Media.Image image) {
        return Lfm3VisionPreprocess.targetSize(
                image.width(), image.height(), patch * MERGE, minPixels, maxPixels);
    }

    @Override
    public int positions(Media.Image image) {
        int[] size = targetSize(image);
        return size[0] / (patch * MERGE) * (size[1] / (patch * MERGE));
    }

    @Override
    public String planId() {
        return "lfm3vl patch=" + patch + " pixels=" + minPixels + ".." + maxPixels;
    }

    @Override
    public void project(Media.Image image, int maxChunkSize, Consumer<MemoryView<?>> sink) {
        if (maxChunkSize <= 0) throw new IllegalArgumentException("maxChunkSize must be positive");
        int rows = positions(image);
        MemoryArena<MemorySegment> scratch = Arenas.newCrossThreadMemoryArena();
        try {
            MemoryView<MemorySegment> rowsOut = encode(image, scratch, null);
            for (int first = 0; first < rows; first += maxChunkSize)
                sink.accept(rowsOut.slice(0, first, Math.min(rows, first + maxChunkSize)));
        } finally {
            Arenas.close(scratch);
            Reference.reachabilityFence(this);
        }
    }

    /** Every buffer of one encode, allocated before the first layer runs. */
    private final class Scratch {
        final int width, height, count;
        final MemoryView<MemorySegment> patches, x, normed, wide, query, key, value, half, mixed;
        final MemoryView<MemorySegment> hidden, hidden2, merged, projected, output, cos, sin;
        final float[] gated, conv; // the depthwise conv's input and output

        Scratch(int width, int height, MemoryArena<MemorySegment> arena) {
            this.width = width;
            this.height = height;
            count = width * height;
            int kvDim = kvHeads * headDim, tokens = count / (MERGE * MERGE);
            patches = Views.allocateF32(arena, count, 3 * patch * patch);
            x = Views.allocateF32(arena, count, dim);
            normed = Views.allocateF32(arena, count, dim);
            wide = Views.allocateF32(arena, count, 3 * dim);
            query = Views.allocateF32(arena, count, dim);
            key = Views.allocateF32(arena, count, kvDim);
            value = Views.allocateF32(arena, count, kvDim);
            half = Views.allocateF16(arena, count, kvDim);
            gated = new float[count * dim];
            conv = new float[count * dim];
            mixed = Views.allocateF32(arena, count, dim);
            hidden = Views.allocateF32(arena, count, ffnDim);
            hidden2 = Views.allocateF32(arena, count, ffnDim);
            merged = Views.allocateF32(arena, tokens, dim * MERGE * MERGE);
            projected = Views.allocateF32(arena, tokens, projectorHidden);
            output = Views.allocateF32(arena, tokens, modelDim);
            // 2D RoPE angles, one row per patch: pairs j < headDim/4 turn by y, the rest by x
            int lanes = headDim / 2, quarter = headDim / 4;
            float[] c = new float[count * lanes], s = new float[count * lanes];
            for (int p = 0; p < count; p++)
                for (int j = 0; j < lanes; j++) {
                    float theta =
                            (j < quarter ? p / width : p % width)
                                    * ropeInverseFrequency[j % quarter];
                    c[p * lanes + j] = (float) Math.cos(theta);
                    s[p * lanes + j] = (float) Math.sin(theta);
                }
            cos = Views.allocateF32(arena, count, lanes);
            sin = Views.allocateF32(arena, count, lanes);
            Views.copyFromArray(cos, 0, c, 0, c.length, "rope cos");
            Views.copyFromArray(sin, 0, s, 0, s.length, "rope sin");
        }
    }

    /**
     * The rows of {@code image}; {@code afterLayer} (nullable) sees the residual after each block.
     */
    MemoryView<MemorySegment> encode(
            Media.Image image,
            MemoryArena<MemorySegment> arena,
            ObjIntConsumer<MemoryView<MemorySegment>> afterLayer) {
        int[] size = targetSize(image);
        int[] rgb = Lfm3VisionPreprocess.resize(image, size[0], size[1]);
        float[] pixels = Lfm3VisionPreprocess.patches(rgb, size[0], size[1], patch, mean, std);
        Scratch s = new Scratch(size[0] / patch, size[1] / patch, arena);
        // ggml_conv_2d's im2col is F16 for any kernel but BF16, and its matmul runs in F16
        if (halfPixels) Towers.roundToHalf(pixels);
        Views.copyFromArray(s.patches, 0, pixels, 0, pixels.length, "patches");
        MatMul.gemm(patchKernel, s.patches, s.x, s.count);
        Norms.rmsnormRowsGgml(s.x, s.x, embeddingNorm, s.count, dim, eps);

        for (int l = 0; l < layers.length; l++) {
            Layer layer = layers[l];
            Norms.rmsnormRowsGgml(s.normed, s.x, layer.norm1, s.count, dim, eps);
            if (layer.convIn != null) conv(layer, s);
            else attention(layer, s);
            Ops.addRows(s.x, s.mixed, s.count, dim);
            Norms.rmsnormRowsGgml(s.normed, s.x, layer.norm2, s.count, dim, eps);
            Towers.ffn(
                    layer.ffnGate,
                    layer.ffnUp,
                    layer.ffnDown,
                    s.normed,
                    s.hidden,
                    s.hidden2,
                    s.mixed,
                    s.count,
                    ffnDim);
            Ops.addRows(s.x, s.mixed, s.count, dim);
            if (afterLayer != null) afterLayer.accept(s.x, l);
        }

        // the projector's LayerNorm takes projector_layer_norm_eps, not the tower's
        Norms.layerNormRows(s.x, s.x, projectorNorm, projectorNormBias, s.count, dim, 1e-6f);
        merge(s);
        int tokens = s.count / (MERGE * MERGE);
        MatMul.gemm(projectorUp, s.merged, s.projected, tokens);
        Ops.addRowBiasInPlace(s.projected, 0, projectorUpBias, 0, tokens, projectorHidden);
        GeluErf.rows(s.projected, tokens, projectorHidden);
        MatMul.gemm(projectorDown, s.projected, s.output, tokens);
        Ops.addRowBiasInPlace(s.output, 0, projectorDownBias, 0, tokens, modelDim);
        return s.output;
    }

    /** B * x through the 3x3 depthwise conv (zero padded), gated by C, out-projected. */
    private void conv(Layer layer, Scratch s) {
        MatMul.gemm(layer.convIn, s.normed, s.wide, s.count);
        float[] gated = s.gated, conv = s.conv, taps = layer.convTaps;
        Parallel.forLoop(
                s.count,
                p -> {
                    long row = (long) p * 3 * dim;
                    Ops.multiplyInPlace(s.wide, row, s.wide, row + 2L * dim, dim);
                    Views.copyToArray(s.wide, row, gated, p * dim, dim, "gated");
                });
        Parallel.forLoop(
                s.height,
                y -> {
                    for (int x = 0; x < s.width; x++) {
                        int o = (y * s.width + x) * dim;
                        Arrays.fill(conv, o, o + dim, 0f);
                        for (int ky = 0; ky < 3; ky++) {
                            int sy = y + ky - 1;
                            if (sy < 0 || sy >= s.height) continue;
                            for (int kx = 0; kx < 3; kx++) {
                                int sx = x + kx - 1;
                                if (sx < 0 || sx >= s.width) continue;
                                int in = (sy * s.width + sx) * dim, tap = (ky * 3 + kx) * dim;
                                for (int c = 0; c < dim; c++)
                                    conv[o + c] += taps[tap + c] * gated[in + c];
                            }
                        }
                    }
                });
        Views.copyFromArray(s.normed, 0, conv, 0, conv.length, "conv");
        Parallel.forLoop(
                s.count,
                p ->
                        Ops.multiplyInPlace(
                                s.normed, (long) p * dim, s.wide, (long) p * 3 * dim + dim, dim));
        MatMul.gemm(layer.convOut, s.normed, s.mixed, s.count);
    }

    private void attention(Layer layer, Scratch s) {
        int kvDim = kvHeads * headDim;
        MatMul.gemm(layer.query, s.normed, s.query, s.count);
        MatMul.gemm(layer.key, s.normed, s.key, s.count);
        MatMul.gemm(layer.value, s.normed, s.value, s.count);
        Parallel.forLoop(
                s.count,
                p -> {
                    // the patch's 2D RoPE pairs (j, j + headDim/2), so NeoX over its table
                    Towers.headNormRope(
                            s.query,
                            (long) p * dim,
                            heads,
                            headDim,
                            layer.queryNorm,
                            eps,
                            p,
                            s.cos,
                            s.sin);
                    Towers.headNormRope(
                            s.key,
                            (long) p * kvDim,
                            kvHeads,
                            headDim,
                            layer.keyNorm,
                            eps,
                            p,
                            s.cos,
                            s.sin);
                });
        // ggml's flash attention reads K and V in F16: round them so, keep them F32 for speed
        Towers.roundToHalf(s.key, s.half, s.count * kvDim);
        Towers.roundToHalf(s.value, s.half, s.count * kvDim);
        FlashAttention.bidirectionalPrefill(
                s.query,
                s.normed,
                s.key,
                s.value,
                heads,
                s.count,
                headDim,
                kvDim,
                dim,
                heads / kvHeads,
                1f / (float) Math.sqrt(headDim));
        MatMul.gemm(layer.output, s.normed, s.mixed, s.count);
    }

    /** 2x2 blocks of the raster grid into one row each: (y, x), (y, x+1), (y+1, x), (y+1, x+1). */
    private void merge(Scratch s) {
        int outX = s.width / MERGE, outY = s.height / MERGE, wide = dim * MERGE * MERGE;
        Parallel.forLoop(
                outY,
                oy -> {
                    for (int ox = 0; ox < outX; ox++)
                        for (int iy = 0; iy < MERGE; iy++)
                            for (int ix = 0; ix < MERGE; ix++)
                                Convert.copyF32(
                                        s.x,
                                        ((long) (oy * MERGE + iy) * s.width + ox * MERGE + ix)
                                                * dim,
                                        s.merged,
                                        ((long) oy * outX + ox) * wide
                                                + (long) (iy * MERGE + ix) * dim,
                                        dim);
                });
    }

    // --- loading ---

    /** A patchifier kernel, {@code [dim][3][patch][patch]} in any float type. */
    private MemoryView<MemorySegment> kernel(
            Map<String, MemoryView<MemorySegment>> tensors, String name, int patchVector) {
        MemoryView<MemorySegment> value = ModelLoader.require(tensors, name);
        require(
                value.shape().size() == (long) dim * patchVector,
                name + " has shape " + value.shape());
        return value;
    }
}
