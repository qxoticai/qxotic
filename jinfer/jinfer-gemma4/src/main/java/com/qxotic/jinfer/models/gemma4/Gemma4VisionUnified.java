package com.qxotic.jinfer.models.gemma4;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.Arenas;
import com.qxotic.jinfer.Parallel;
import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.kernels.MatMul;
import com.qxotic.jinfer.kernels.ModelLoader;
import com.qxotic.jinfer.kernels.Norms;
import com.qxotic.jinfer.kernels.Ops;
import com.qxotic.jinfer.media.Media;
import com.qxotic.jinfer.media.MediaProjector;
import com.qxotic.jota.DataType;
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

/** Gemma 4 unified vision projector ({@code projector_type=gemma4uv}). */
public final class Gemma4VisionUnified implements MediaProjector<Media.Image> {
    static final float LAYER_NORM_EPS = 1e-5f;

    private final int patchSize, visionDim, modelDim, positionSize, patchVector;
    private final float rmsEps;
    private final MemoryView<MemorySegment> patchEmbedding,
            patchBias,
            positionEmbedding,
            inputProjection,
            norm1Weight,
            norm1Bias,
            norm2Weight,
            norm2Bias,
            norm3Weight,
            norm3Bias;

    Gemma4VisionUnified(
            int patchSize,
            int visionDim,
            int modelDim,
            int positionSize,
            float rmsEps,
            MemoryView<MemorySegment> patchEmbedding,
            MemoryView<MemorySegment> patchBias,
            MemoryView<MemorySegment> positionEmbedding,
            MemoryView<MemorySegment> inputProjection,
            MemoryView<MemorySegment> norm1Weight,
            MemoryView<MemorySegment> norm1Bias,
            MemoryView<MemorySegment> norm2Weight,
            MemoryView<MemorySegment> norm2Bias,
            MemoryView<MemorySegment> norm3Weight,
            MemoryView<MemorySegment> norm3Bias) {
        if (patchSize <= 0 || visionDim <= 0 || modelDim <= 0 || positionSize <= 0)
            throw new IllegalArgumentException("vision dimensions must be positive");
        this.patchSize = patchSize;
        this.visionDim = visionDim;
        this.modelDim = modelDim;
        this.positionSize = positionSize;
        this.patchVector = Math.multiplyExact(3, Math.multiplyExact(patchSize, patchSize));
        this.rmsEps = rmsEps;
        this.patchEmbedding =
                requirePatchWeight(
                        patchEmbedding, "v.patch_embd.weight", Shape.flat(visionDim, patchVector));
        this.patchBias = requireF32(patchBias, "v.patch_embd.bias", Shape.flat(visionDim));
        this.positionEmbedding =
                requireF32(
                        positionEmbedding,
                        "v.position_embd.weight",
                        Shape.flat(2, positionSize, visionDim));
        this.inputProjection =
                requireWeight(
                        inputProjection,
                        "mm.input_projection.weight",
                        Shape.flat(modelDim, visionDim));
        this.norm1Weight =
                requireF32(norm1Weight, "v.patch_norm.1.weight", Shape.flat(patchVector));
        this.norm1Bias = requireF32(norm1Bias, "v.patch_norm.1.bias", Shape.flat(patchVector));
        this.norm2Weight = requireF32(norm2Weight, "v.patch_norm.2.weight", Shape.flat(visionDim));
        this.norm2Bias = requireF32(norm2Bias, "v.patch_norm.2.bias", Shape.flat(visionDim));
        this.norm3Weight = requireF32(norm3Weight, "v.patch_norm.3.weight", Shape.flat(visionDim));
        this.norm3Bias = requireF32(norm3Bias, "v.patch_norm.3.bias", Shape.flat(visionDim));
    }

    @Override
    public int positions(Media.Image image) {
        int[] size = targetSize(image, VisionPreprocess.budget(280));
        return Math.multiplyExact(size[0] / patchSize, size[1] / patchSize);
    }

    @Override
    public String planId() {
        return "gemma4uv patch=" + patchSize + " positions=" + positionSize;
    }

    @Override
    public void project(Media.Image image, int maxChunkSize, Consumer<MemoryView<?>> sink) {
        Objects.requireNonNull(image, "image");
        Objects.requireNonNull(sink, "sink");
        if (maxChunkSize <= 0) throw new IllegalArgumentException("maxChunkSize must be positive");
        int[] size = targetSize(image, VisionPreprocess.budget(280));
        int count = Math.multiplyExact(size[0] / patchSize, size[1] / patchSize);
        if (count > maxChunkSize)
            throw new IllegalArgumentException(
                    "vision block has "
                            + count
                            + " projected rows, exceeding maxChunkSize "
                            + maxChunkSize);
        MemoryArena<MemorySegment> scratch = Arenas.newCrossThreadMemoryArena();
        try {
            MemoryView<MemorySegment> rows = encode(image, scratch, size);
            sink.accept(rows);
        } finally {
            Arenas.close(scratch);
        }
    }

    private MemoryView<MemorySegment> encode(
            Media.Image image, MemoryArena<MemorySegment> scratch, int[] size) {
        int targetWidth = size[0], targetHeight = size[1];
        int patchesX = targetWidth / patchSize;
        int patchesY = targetHeight / patchSize;
        if (patchesX > positionSize || patchesY > positionSize)
            throw new IllegalArgumentException(
                    "patch grid "
                            + patchesX
                            + "x"
                            + patchesY
                            + " exceeds position table "
                            + positionSize);
        int count = Math.multiplyExact(patchesX, patchesY);

        MemoryView<MemorySegment> flat =
                VisionPreprocess.im2col(image, targetWidth, targetHeight, patchSize, scratch);
        Norms.layerNorm(flat, flat, norm1Weight, norm1Bias, patchVector, count, LAYER_NORM_EPS);

        MemoryView<MemorySegment> current = Views.allocateF32(scratch, count, visionDim);
        MatMul.gemm(patchEmbedding, flat, current, count);
        Ops.addRowBiasInPlace(current, 0, patchBias, 0, count, visionDim);
        Norms.layerNorm(current, current, norm2Weight, norm2Bias, visionDim, count, LAYER_NORM_EPS);

        Ops.addGridPositions(current, positionEmbedding, count, patchesX, visionDim, positionSize);
        Norms.layerNorm(current, current, norm3Weight, norm3Bias, visionDim, count, LAYER_NORM_EPS);
        Parallel.forRows(
                count,
                row ->
                        Norms.rmsnormNoWeight(
                                current,
                                (long) row * visionDim,
                                current,
                                (long) row * visionDim,
                                visionDim,
                                rmsEps));

        MemoryView<MemorySegment> projected = Views.allocateF32(scratch, count, modelDim);
        MatMul.gemm(inputProjection, current, projected, count);
        return projected;
    }

    private int[] targetSize(Media.Image image, int budgetTokens) {
        if (!VisionPreprocess.SMART_RESIZE) return new int[] {16 * patchSize, 16 * patchSize};
        int patchArea = Math.multiplyExact(patchSize, patchSize);
        return VisionPreprocess.smartResize(
                image.width(),
                image.height(),
                patchSize,
                Math.multiplyExact(40, patchArea),
                Math.multiplyExact(budgetTokens, patchArea));
    }

    public static Gemma4VisionUnified loadModel(Path path, Arena arena) throws IOException {
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            GGUF gguf = ModelLoader.readGguf(channel, path.toString());
            return loadModel(path, gguf, ModelLoader.loadTensors(channel, gguf, arena));
        }
    }

    public static Gemma4VisionUnified loadModel(
            GGUF gguf, Map<String, MemoryView<MemorySegment>> tensors) {
        return loadModel(Path.of("mmproj.gguf"), gguf, tensors);
    }

    public static Gemma4VisionUnified loadModel(
            Path label, GGUF gguf, Map<String, MemoryView<MemorySegment>> tensors) {
        Objects.requireNonNull(label, "label");
        Objects.requireNonNull(gguf, "gguf");
        Objects.requireNonNull(tensors, "tensors");
        int basePatch = gguf.getValueOrDefault(int.class, "clip.vision.patch_size", 16);
        int merge = gguf.getValueOrDefault(int.class, "clip.vision.proj_scale_factor", 3);
        int patchSize = Math.multiplyExact(basePatch, Math.max(1, merge));
        int visionDim = gguf.getValueOrDefault(int.class, "clip.vision.embedding_length", 3840);
        int modelDim = gguf.getValueOrDefault(int.class, "clip.vision.projection_dim", visionDim);
        float rmsEps =
                gguf.getValueOrDefault(
                        float.class, "clip.vision.attention.layer_norm_epsilon", 1e-6f);
        MemoryView<MemorySegment> position = require(tensors, "v.position_embd.weight");
        Shape positionShape = position.dataType().logicalShape(position.shape());
        if (position.dataType() != DataType.FP32
                || !positionShape.isFlat()
                || positionShape.flatRank() != 3
                || positionShape.flatAt(0) != 2
                || positionShape.flatAt(2) != visionDim)
            throw new IllegalArgumentException(
                    label.getFileName()
                            + ": v.position_embd.weight expected shape [2, positions, "
                            + visionDim
                            + "] FP32 but was "
                            + positionShape
                            + " "
                            + position.dataType().name());
        int positionSize = Math.toIntExact(positionShape.flatAt(1));
        return new Gemma4VisionUnified(
                patchSize,
                visionDim,
                modelDim,
                positionSize,
                rmsEps,
                require(tensors, "v.patch_embd.weight"),
                require(tensors, "v.patch_embd.bias"),
                position,
                require(tensors, "mm.input_projection.weight"),
                require(tensors, "v.patch_norm.1.weight"),
                require(tensors, "v.patch_norm.1.bias"),
                require(tensors, "v.patch_norm.2.weight"),
                require(tensors, "v.patch_norm.2.bias"),
                require(tensors, "v.patch_norm.3.weight"),
                require(tensors, "v.patch_norm.3.bias"));
    }

    private static MemoryView<MemorySegment> require(
            Map<String, MemoryView<MemorySegment>> tensors, String name) {
        MemoryView<MemorySegment> value = tensors.get(name);
        if (value == null) throw new IllegalStateException("mmproj tensor missing: " + name);
        return value;
    }

    static MemoryView<MemorySegment> requireWeight(
            Map<String, MemoryView<MemorySegment>> tensors, String name, Shape expected) {
        return requireWeight(require(tensors, name), name, expected);
    }

    static MemoryView<MemorySegment> requireWeight(
            MemoryView<MemorySegment> value, String name, Shape expected) {
        Views.requireContiguous(value, name);
        DataType type = value.dataType();
        if (type != DataType.FP32 && type != DataType.BF16 && type != DataType.Q8_0)
            throw new IllegalArgumentException(name + ": unsupported weight type " + type.name());
        Shape actual = type.logicalShape(value.shape());
        if (!actual.equals(expected))
            throw new IllegalArgumentException(
                    name + ": expected shape " + expected + " but was " + actual);
        return value;
    }

    static MemoryView<MemorySegment> requirePatchWeight(
            MemoryView<MemorySegment> value, String name, Shape expected) {
        Views.requireContiguous(value, name);
        DataType type = value.dataType();
        if (type != DataType.FP32 && type != DataType.BF16 && type != DataType.Q8_0)
            throw new IllegalArgumentException(name + ": unsupported weight type " + type.name());
        Shape actual = type.logicalShape(value.shape());
        if (!actual.isFlat()
                || actual.flatAt(0) != expected.flatAt(0)
                || actual.size() != expected.size())
            throw new IllegalArgumentException(
                    name + ": expected output/input shape " + expected + " but was " + actual);
        return value;
    }

    static MemoryView<MemorySegment> requireF32(
            MemoryView<MemorySegment> value, String name, Shape expected) {
        Objects.requireNonNull(value, name);
        Views.requireDense(value, DataType.FP32, name);
        if (!value.shape().equals(expected))
            throw new IllegalArgumentException(
                    name + ": expected shape " + expected + " but was " + value.shape());
        return value;
    }
}
