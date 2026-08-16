package com.qxotic.jinfer.models.gemma4;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.Parallel;
import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.boundary.Arenas;
import com.qxotic.jinfer.boundary.Media;
import com.qxotic.jinfer.boundary.MediaProjector;
import com.qxotic.jinfer.kernels.Activations;
import com.qxotic.jinfer.kernels.Convert;
import com.qxotic.jinfer.kernels.FlashAttention;
import com.qxotic.jinfer.kernels.MatMul;
import com.qxotic.jinfer.kernels.ModelLoader;
import com.qxotic.jinfer.kernels.Norms;
import com.qxotic.jinfer.kernels.Ops;
import com.qxotic.jinfer.kernels.RoPE;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryArena;
import com.qxotic.jota.memory.MemoryView;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Map;
import java.util.Objects;
import java.util.function.Consumer;

/** Gemma 4 SigLIP-style vision tower ({@code projector_type=gemma4v}). */
public final class Gemma4Vision implements MediaProjector<Media.Image> {
    private static final float ROPE_THETA = 100f;

    private final int imageSize,
            patchSize,
            visionDim,
            headCount,
            headDim,
            ffnDim,
            modelDim,
            merge,
            positionSize;
    private final float normEps;
    private final MemoryView<MemorySegment> patchEmbedding, positionEmbedding;
    private final Clamped inputProjection;
    private final Layer[] layers;

    record Layer(
            MemoryView<MemorySegment> norm1,
            MemoryView<MemorySegment> norm2,
            MemoryView<MemorySegment> attentionPostNorm,
            MemoryView<MemorySegment> ffnPostNorm,
            MemoryView<MemorySegment> queryNorm,
            MemoryView<MemorySegment> keyNorm,
            Clamped query,
            Clamped key,
            Clamped value,
            Clamped attentionOutput,
            Clamped gate,
            Clamped up,
            Clamped down) {}

    Gemma4Vision(
            int imageSize,
            int patchSize,
            int visionDim,
            int headCount,
            int ffnDim,
            int modelDim,
            int merge,
            int positionSize,
            float normEps,
            MemoryView<MemorySegment> patchEmbedding,
            MemoryView<MemorySegment> positionEmbedding,
            Clamped inputProjection,
            Layer[] layers) {
        if (imageSize <= 0
                || patchSize <= 0
                || visionDim <= 0
                || headCount <= 0
                || ffnDim <= 0
                || modelDim <= 0
                || merge <= 0
                || positionSize <= 0)
            throw new IllegalArgumentException("vision dimensions must be positive");
        if (visionDim % headCount != 0)
            throw new IllegalArgumentException(
                    "head_count " + headCount + " does not divide embedding_length " + visionDim);
        if (imageSize % patchSize != 0)
            throw new IllegalArgumentException("image_size must be divisible by patch_size");
        this.headDim = visionDim / headCount;
        if (headDim % 4 != 0)
            throw new IllegalArgumentException(
                    "vision head dimension " + headDim + " must be divisible by 4 for 2D RoPE");
        this.imageSize = imageSize;
        this.patchSize = patchSize;
        this.visionDim = visionDim;
        this.headCount = headCount;
        this.ffnDim = ffnDim;
        this.modelDim = modelDim;
        this.merge = merge;
        this.positionSize = positionSize;
        this.normEps = normEps;
        int patchVector = Math.multiplyExact(3, Math.multiplyExact(patchSize, patchSize));
        this.patchEmbedding =
                Gemma4VisionUnified.requirePatchWeight(
                        patchEmbedding, "v.patch_embd.weight", Shape.flat(visionDim, patchVector));
        this.positionEmbedding =
                Gemma4VisionUnified.requireF32(
                        positionEmbedding,
                        "v.position_embd.weight",
                        Shape.flat(2, positionSize, visionDim));
        this.inputProjection =
                requireClamped(inputProjection, "mm.input_projection", modelDim, visionDim);
        this.layers = Objects.requireNonNull(layers, "layers").clone();
        for (int i = 0; i < this.layers.length; i++) validateLayer(this.layers[i], i);
    }

    @Override
    public int positions(Media.Image image) {
        Objects.requireNonNull(image, "image");
        int[] size = targetSize(image, VisionPreprocess.budget(280));
        int patchesX = size[0] / patchSize, patchesY = size[1] / patchSize;
        return Math.multiplyExact(Math.max(1, patchesX / merge), Math.max(1, patchesY / merge));
    }

    @Override
    public String planId() {
        return "gemma4v patch="
                + patchSize
                + " merge="
                + merge
                + " imageSize="
                + imageSize
                + " positions="
                + positionSize;
    }

    @Override
    public void project(Media.Image image, int maxChunkSize, Consumer<MemoryView<?>> sink) {
        Objects.requireNonNull(image, "image");
        Objects.requireNonNull(sink, "sink");
        if (maxChunkSize <= 0) throw new IllegalArgumentException("maxChunkSize must be positive");
        int[] size = targetSize(image, VisionPreprocess.budget(280));
        int patchesX = size[0] / patchSize, patchesY = size[1] / patchSize;
        int rows = Math.multiplyExact(Math.max(1, patchesX / merge), Math.max(1, patchesY / merge));
        if (rows > maxChunkSize)
            throw new IllegalArgumentException(
                    "vision block has "
                            + rows
                            + " projected rows, exceeding maxChunkSize "
                            + maxChunkSize);
        MemoryArena<MemorySegment> scratch = Arenas.newCrossThreadMemoryArena();
        try {
            sink.accept(encode(image, scratch, size));
        } finally {
            Arenas.close(scratch);
        }
    }

    private MemoryView<MemorySegment> encode(
            Media.Image image, MemoryArena<MemorySegment> scratch, int[] size) {
        int patchesX = size[0] / patchSize, patchesY = size[1] / patchSize;
        if (patchesX > positionSize || patchesY > positionSize)
            throw new IllegalArgumentException(
                    "patch grid "
                            + patchesX
                            + "x"
                            + patchesY
                            + " exceeds position table "
                            + positionSize);
        int count = Math.multiplyExact(patchesX, patchesY);
        int patchVector = Math.multiplyExact(3, Math.multiplyExact(patchSize, patchSize));
        MemoryView<MemorySegment> flat =
                VisionPreprocess.im2col(image, size[0], size[1], patchSize, scratch);
        MemoryView<MemorySegment> current = Views.allocateF32(scratch, count, visionDim);
        MatMul.gemm(
                patchEmbedding,
                flat,
                patchVector,
                current,
                visionDim,
                visionDim,
                count,
                patchVector);
        addPositions(current, count, patchesX);

        MemoryView<MemorySegment> normalized = Views.allocateF32(scratch, count, visionDim);
        MemoryView<MemorySegment> temporary = Views.allocateF32(scratch, count, visionDim);
        MemoryView<MemorySegment> query = Views.allocateF32(scratch, count, visionDim);
        MemoryView<MemorySegment> key = Views.allocateF32(scratch, count, visionDim);
        MemoryView<MemorySegment> value = Views.allocateF32(scratch, count, visionDim);
        MemoryView<MemorySegment> attention = Views.allocateF32(scratch, count, visionDim);
        MemoryView<MemorySegment> gate = Views.allocateF32(scratch, count, ffnDim);
        MemoryView<MemorySegment> up = Views.allocateF32(scratch, count, ffnDim);
        MemoryView<MemorySegment> keyF16 = Views.allocateF16(scratch, count, visionDim);
        MemoryView<MemorySegment> valueF16 = Views.allocateF16(scratch, count, visionDim);
        MemoryView<MemorySegment> clampScratch =
                Views.allocateF32(scratch, count, Math.max(visionDim, ffnDim));

        for (Layer layer : layers) {
            Norms.rmsnormRows(normalized, current, layer.norm1(), count, visionDim, normEps);
            attention(
                    normalized,
                    query,
                    key,
                    value,
                    attention,
                    keyF16,
                    valueF16,
                    layer,
                    patchesX,
                    count,
                    clampScratch);
            postNormResidual(current, attention, layer.attentionPostNorm(), temporary, count);
            Norms.rmsnormRows(normalized, current, layer.norm2(), count, visionDim, normEps);
            layer.gate().gemm(normalized, visionDim, gate, ffnDim, count, clampScratch);
            layer.up().gemm(normalized, visionDim, up, ffnDim, count, clampScratch);
            Parallel.forRows(
                    count,
                    row ->
                            Activations.quickGeluMultiply(
                                    gate, (long) row * ffnDim, up, (long) row * ffnDim, ffnDim));
            layer.down().gemm(gate, ffnDim, attention, visionDim, count, clampScratch);
            postNormResidual(current, attention, layer.ffnPostNorm(), temporary, count);
        }
        return projectPooled(current, patchesX, patchesY, scratch, clampScratch);
    }

    private void attention(
            MemoryView<MemorySegment> x,
            MemoryView<MemorySegment> query,
            MemoryView<MemorySegment> key,
            MemoryView<MemorySegment> value,
            MemoryView<MemorySegment> output,
            MemoryView<MemorySegment> keyF16,
            MemoryView<MemorySegment> valueF16,
            Layer layer,
            int patchesX,
            int count,
            MemoryView<MemorySegment> clampScratch) {
        layer.query().gemm(x, visionDim, query, visionDim, count, clampScratch);
        layer.key().gemm(x, visionDim, key, visionDim, count, clampScratch);
        layer.value().gemm(x, visionDim, value, visionDim, count, clampScratch);
        Parallel.forRows(
                count,
                token -> {
                    for (int head = 0; head < headCount; head++) {
                        long offset = (long) token * visionDim + (long) head * headDim;
                        Norms.rmsnorm(
                                query, offset, query, offset, layer.queryNorm(), headDim, normEps);
                        Norms.rmsnorm(key, offset, key, offset, layer.keyNorm(), headDim, normEps);
                        Norms.rmsnormNoWeight(value, offset, value, offset, headDim, normEps);
                        rope2d(
                                query,
                                offset,
                                headDim,
                                token % patchesX,
                                token / patchesX,
                                ROPE_THETA);
                        rope2d(
                                key,
                                offset,
                                headDim,
                                token % patchesX,
                                token / patchesX,
                                ROPE_THETA);
                    }
                });
        int elements = Math.multiplyExact(count, visionDim);
        Convert.f32ToF16(key, 0, keyF16, 0, elements);
        Convert.f32ToF16(value, 0, valueF16, 0, elements);
        FlashAttention.bidirectionalPrefill(
                query, output, keyF16, valueF16, headCount, count, headDim, visionDim, visionDim, 1,
                1f);
        layer.attentionOutput().gemm(output, visionDim, query, visionDim, count, clampScratch);
        Convert.copyF32(query, 0, output, 0, elements);
    }

    private void postNormResidual(
            MemoryView<MemorySegment> residual,
            MemoryView<MemorySegment> value,
            MemoryView<MemorySegment> weight,
            MemoryView<MemorySegment> temporary,
            int rows) {
        Norms.rmsnormRows(temporary, value, weight, rows, visionDim, normEps);
        Ops.addInPlace(residual, 0, temporary, 0, Math.multiplyExact(rows, visionDim));
    }

    private MemoryView<MemorySegment> projectPooled(
            MemoryView<MemorySegment> current,
            int patchesX,
            int patchesY,
            MemoryArena<MemorySegment> scratch,
            MemoryView<MemorySegment> clampScratch) {
        int outputX = Math.max(1, patchesX / merge), outputY = Math.max(1, patchesY / merge);
        int rows = Math.multiplyExact(outputX, outputY);
        MemoryView<MemorySegment> pooled = Views.allocateF32(scratch, rows, visionDim);
        float scale = (float) Math.sqrt(visionDim);
        Ops.windowedMeanPool(current, patchesX, patchesY, merge, visionDim, pooled);
        Parallel.forRows(
                rows,
                row -> {
                    long destinationBase = (long) row * visionDim;
                    Ops.mapInPlace(pooled, destinationBase, visionDim, value -> value * scale);
                    Norms.rmsnormNoWeight(
                            pooled, destinationBase, pooled, destinationBase, visionDim, normEps);
                });
        MemoryView<MemorySegment> projected = Views.allocateF32(scratch, rows, modelDim);
        inputProjection.gemm(pooled, visionDim, projected, modelDim, rows, clampScratch);
        return projected;
    }

    private void addPositions(MemoryView<MemorySegment> current, int count, int patchesX) {
        Ops.addGridPositions(current, positionEmbedding, count, patchesX, visionDim, positionSize);
    }

    static void rope2d(
            MemoryView<MemorySegment> value,
            long base,
            int headDim,
            int positionX,
            int positionY,
            float theta) {
        if (headDim <= 0 || headDim % 4 != 0 || theta <= 0f)
            throw new IllegalArgumentException("invalid 2D RoPE geometry");
        int half = headDim / 2;
        RoPE.rotatePairs(value, base, half / 2, half, positionX, theta);
        RoPE.rotatePairs(value, base + half, half / 2, half, positionY, theta);
    }

    private int[] targetSize(Media.Image image, int budgetTokens) {
        if (!VisionPreprocess.SMART_RESIZE) return new int[] {imageSize, imageSize};
        int factor = Math.multiplyExact(patchSize, merge);
        int area = Math.multiplyExact(factor, factor);
        return VisionPreprocess.smartResize(
                image.width(),
                image.height(),
                factor,
                area,
                Math.multiplyExact(budgetTokens, area));
    }

    private void validateLayer(Layer layer, int index) {
        Objects.requireNonNull(layer, "layer " + index);
        String prefix = "v.blk." + index + ".";
        Gemma4VisionUnified.requireF32(layer.norm1(), prefix + "ln1.weight", Shape.flat(visionDim));
        Gemma4VisionUnified.requireF32(layer.norm2(), prefix + "ln2.weight", Shape.flat(visionDim));
        Gemma4VisionUnified.requireF32(
                layer.attentionPostNorm(), prefix + "attn_post_norm.weight", Shape.flat(visionDim));
        Gemma4VisionUnified.requireF32(
                layer.ffnPostNorm(), prefix + "ffn_post_norm.weight", Shape.flat(visionDim));
        Gemma4VisionUnified.requireF32(
                layer.queryNorm(), prefix + "attn_q_norm.weight", Shape.flat(headDim));
        Gemma4VisionUnified.requireF32(
                layer.keyNorm(), prefix + "attn_k_norm.weight", Shape.flat(headDim));
        requireClamped(layer.query(), prefix + "attn_q", visionDim, visionDim);
        requireClamped(layer.key(), prefix + "attn_k", visionDim, visionDim);
        requireClamped(layer.value(), prefix + "attn_v", visionDim, visionDim);
        requireClamped(layer.attentionOutput(), prefix + "attn_out", visionDim, visionDim);
        requireClamped(layer.gate(), prefix + "ffn_gate", ffnDim, visionDim);
        requireClamped(layer.up(), prefix + "ffn_up", ffnDim, visionDim);
        requireClamped(layer.down(), prefix + "ffn_down", visionDim, ffnDim);
    }

    private static Clamped requireClamped(
            Clamped clamped, String name, int outputDim, int inputDim) {
        Objects.requireNonNull(clamped, name);
        Gemma4VisionUnified.requireWeight(
                clamped.weight(), name + ".weight", Shape.flat(outputDim, inputDim));
        if (clamped.inputMin() > clamped.inputMax() || clamped.outputMin() > clamped.outputMax())
            throw new IllegalArgumentException(name + ": invalid clamp bounds");
        return clamped;
    }

    public static Gemma4Vision loadModel(Path path, Arena arena) throws IOException {
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            GGUF gguf = ModelLoader.readGguf(channel, path.toString());
            return loadModel(path, gguf, ModelLoader.loadTensors(channel, gguf, arena));
        }
    }

    public static Gemma4Vision loadModel(
            GGUF gguf, Map<String, MemoryView<MemorySegment>> tensors) {
        return loadModel(Path.of("mmproj.gguf"), gguf, tensors);
    }

    public static Gemma4Vision loadModel(
            Path label, GGUF gguf, Map<String, MemoryView<MemorySegment>> tensors) {
        Objects.requireNonNull(label, "label");
        Objects.requireNonNull(gguf, "gguf");
        Objects.requireNonNull(tensors, "tensors");
        int patchSize = gguf.getValueOrDefault(int.class, "clip.vision.patch_size", 16);
        int visionDim = gguf.getValueOrDefault(int.class, "clip.vision.embedding_length", 768);
        int headCount = gguf.getValueOrDefault(int.class, "clip.vision.attention.head_count", 12);
        int layerCount = gguf.getValueOrDefault(int.class, "clip.vision.block_count", 16);
        int ffnDim = gguf.getValueOrDefault(int.class, "clip.vision.feed_forward_length", 3072);
        int merge =
                Math.max(1, gguf.getValueOrDefault(int.class, "clip.vision.proj_scale_factor", 3));
        int modelDim = gguf.getValueOrDefault(int.class, "clip.vision.projection_dim", 1536);
        float normEps =
                gguf.getValueOrDefault(
                        float.class, "clip.vision.attention.layer_norm_epsilon", 1e-6f);
        if (patchSize <= 0
                || visionDim <= 0
                || headCount <= 0
                || layerCount < 0
                || ffnDim <= 0
                || modelDim <= 0)
            throw new IllegalArgumentException(label.getFileName() + ": invalid vision metadata");
        int factor = Math.multiplyExact(patchSize, merge);
        int maxPixels =
                VisionPreprocess.IMAGE_TOKEN_BUDGET > 0
                        ? Math.multiplyExact(
                                VisionPreprocess.IMAGE_TOKEN_BUDGET,
                                Math.multiplyExact(factor, factor))
                        : gguf.getValueOrDefault(
                                int.class,
                                "clip.vision.image_max_pixels",
                                Math.multiplyExact(280, Math.multiplyExact(factor, factor)));
        int imageSize = (int) (Math.sqrt(maxPixels) / factor) * factor;
        MemoryView<MemorySegment> position = require(tensors, "v.position_embd.weight");
        if (position.dataType() != com.qxotic.jota.DataType.FP32)
            throw new IllegalArgumentException(
                    label.getFileName() + ": v.position_embd.weight must be FP32");
        Shape positionShape = position.shape();
        if (!positionShape.isFlat()
                || positionShape.flatRank() != 3
                || positionShape.flatAt(0) != 2
                || positionShape.flatAt(2) != visionDim)
            throw new IllegalArgumentException(
                    label.getFileName()
                            + ": v.position_embd.weight expected shape [2, positions, "
                            + visionDim
                            + "] but was "
                            + positionShape);
        int positionSize = Math.toIntExact(positionShape.flatAt(1));
        Layer[] layers = new Layer[layerCount];
        int headDim = visionDim % headCount == 0 ? visionDim / headCount : 0;
        for (int i = 0; i < layerCount; i++) {
            String prefix = "v.blk." + i + ".";
            layers[i] =
                    new Layer(
                            require(tensors, prefix + "ln1.weight"),
                            require(tensors, prefix + "ln2.weight"),
                            require(tensors, prefix + "attn_post_norm.weight"),
                            require(tensors, prefix + "ffn_post_norm.weight"),
                            require(tensors, prefix + "attn_q_norm.weight"),
                            require(tensors, prefix + "attn_k_norm.weight"),
                            Clamped.load(tensors, prefix + "attn_q", visionDim, visionDim),
                            Clamped.load(tensors, prefix + "attn_k", visionDim, visionDim),
                            Clamped.load(tensors, prefix + "attn_v", visionDim, visionDim),
                            Clamped.load(tensors, prefix + "attn_out", visionDim, visionDim),
                            Clamped.load(tensors, prefix + "ffn_gate", ffnDim, visionDim),
                            Clamped.load(tensors, prefix + "ffn_up", ffnDim, visionDim),
                            Clamped.load(tensors, prefix + "ffn_down", visionDim, ffnDim));
        }
        if (headDim == 0)
            throw new IllegalArgumentException(
                    label.getFileName()
                            + ": head_count "
                            + headCount
                            + " does not divide embedding_length "
                            + visionDim);
        return new Gemma4Vision(
                imageSize,
                patchSize,
                visionDim,
                headCount,
                ffnDim,
                modelDim,
                merge,
                positionSize,
                normEps,
                require(tensors, "v.patch_embd.weight"),
                position,
                Clamped.load(tensors, "mm.input_projection", modelDim, visionDim),
                layers);
    }

    private static MemoryView<MemorySegment> require(
            Map<String, MemoryView<MemorySegment>> tensors, String name) {
        MemoryView<MemorySegment> value = tensors.get(name);
        if (value == null) throw new IllegalStateException("mmproj tensor missing: " + name);
        return value;
    }
}
