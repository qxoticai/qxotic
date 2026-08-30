package com.qxotic.jinfer.models.lfm2;

import static com.qxotic.jinfer.kernels.ModelLoader.require;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.Arenas;
import com.qxotic.jinfer.Parallel;
import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.kernels.Activations;
import com.qxotic.jinfer.kernels.Convert;
import com.qxotic.jinfer.kernels.FlashAttention;
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
import java.lang.ref.Reference;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.Map;
import java.util.Objects;
import java.util.function.Consumer;

/** LFM2.5 SigLIP2-NaFlex vision tower and pixel-unshuffle projector. */
public final class Lfm2Vision implements MediaProjector<Media.Image> {
    private final int patchSize,
            visionDim,
            headCount,
            headDim,
            ffnDim,
            merge,
            projectorDim,
            modelDim,
            positionSide;
    private final float normEps;
    private final Lfm2VisionPreprocess.Options preprocessing;
    private final float[] imageMean, imageStd;
    private final MemoryView<MemorySegment> patchWeight, patchBias;
    private final float[] positionEmbedding;
    private final MemoryView<MemorySegment> postNormWeight, postNormBias;
    private final MemoryView<MemorySegment> projectorNormWeight, projectorNormBias;
    private final Linear projectorUp, projectorDown;
    private final Layer[] layers;

    record Linear(
            MemoryView<MemorySegment> weight,
            MemoryView<MemorySegment> bias,
            int outputDim,
            int inputDim) {}

    record Layer(
            MemoryView<MemorySegment> norm1Weight,
            MemoryView<MemorySegment> norm1Bias,
            Linear query,
            Linear key,
            Linear value,
            Linear attentionOutput,
            MemoryView<MemorySegment> norm2Weight,
            MemoryView<MemorySegment> norm2Bias,
            Linear ffnUp,
            Linear ffnDown) {}

    Lfm2Vision(
            int patchSize,
            int visionDim,
            int headCount,
            int ffnDim,
            int merge,
            int projectorDim,
            int modelDim,
            int positionSide,
            float normEps,
            Lfm2VisionPreprocess.Options preprocessing,
            float[] imageMean,
            float[] imageStd,
            MemoryView<MemorySegment> patchWeight,
            MemoryView<MemorySegment> patchBias,
            float[] positionEmbedding,
            MemoryView<MemorySegment> postNormWeight,
            MemoryView<MemorySegment> postNormBias,
            MemoryView<MemorySegment> projectorNormWeight,
            MemoryView<MemorySegment> projectorNormBias,
            Linear projectorUp,
            Linear projectorDown,
            Layer[] layers) {
        if (patchSize <= 0
                || visionDim <= 0
                || headCount <= 0
                || ffnDim <= 0
                || merge <= 1
                || projectorDim <= 0
                || modelDim <= 0
                || positionSide <= 0)
            throw new IllegalArgumentException("vision dimensions must be positive");
        if (!(normEps > 0f) || !Float.isFinite(normEps))
            throw new IllegalArgumentException(
                    "vision LayerNorm epsilon must be finite and positive");
        if (visionDim % headCount != 0)
            throw new IllegalArgumentException(
                    "head_count " + headCount + " does not divide embedding_length " + visionDim);
        this.patchSize = patchSize;
        this.visionDim = visionDim;
        this.headCount = headCount;
        this.headDim = visionDim / headCount;
        this.ffnDim = ffnDim;
        this.merge = merge;
        this.projectorDim = projectorDim;
        this.modelDim = modelDim;
        this.positionSide = positionSide;
        this.normEps = normEps;
        this.preprocessing = Objects.requireNonNull(preprocessing, "preprocessing");
        this.imageMean = requireNormalization(imageMean, "imageMean", false);
        this.imageStd = requireNormalization(imageStd, "imageStd", true);
        if (preprocessing.tileSize() % Math.multiplyExact(patchSize, merge) != 0)
            throw new IllegalArgumentException(
                    "vision tile size must be divisible by patchSize * merge");
        int mergeArea = Math.multiplyExact(merge, merge);
        this.patchWeight =
                requirePatchWeight(
                        patchWeight,
                        "v.patch_embd.weight",
                        visionDim,
                        Math.multiplyExact(3, Math.multiplyExact(patchSize, patchSize)));
        this.patchBias = requireF32(patchBias, "v.patch_embd.bias", Shape.flat(visionDim));
        this.positionEmbedding =
                Objects.requireNonNull(positionEmbedding, "positionEmbedding").clone();
        if (this.positionEmbedding.length
                != Math.multiplyExact(Math.multiplyExact(positionSide, positionSide), visionDim))
            throw new IllegalArgumentException("invalid position embedding size");
        this.postNormWeight = requireF32(postNormWeight, "v.post_ln.weight", Shape.flat(visionDim));
        this.postNormBias = requireF32(postNormBias, "v.post_ln.bias", Shape.flat(visionDim));
        this.projectorNormWeight =
                projectorNormWeight == null
                        ? null
                        : requireF32(
                                projectorNormWeight,
                                "mm.input_norm.weight",
                                Shape.flat(Math.multiplyExact(visionDim, mergeArea)));
        this.projectorNormBias =
                projectorNormBias == null
                        ? null
                        : requireF32(
                                projectorNormBias,
                                "mm.input_norm.bias",
                                Shape.flat(Math.multiplyExact(visionDim, mergeArea)));
        if ((this.projectorNormWeight == null) != (this.projectorNormBias == null))
            throw new IllegalArgumentException("projector LayerNorm requires both weight and bias");
        this.projectorUp =
                requireLinear(
                        projectorUp,
                        "mm.1",
                        projectorDim,
                        Math.multiplyExact(visionDim, mergeArea));
        this.projectorDown = requireLinear(projectorDown, "mm.2", modelDim, projectorDim);
        this.layers = Objects.requireNonNull(layers, "layers").clone();
        for (int i = 0; i < this.layers.length; i++) validateLayer(this.layers[i], i);
    }

    @Override
    public int positions(Media.Image image) {
        return Lfm2VisionPreprocess.positions(
                Objects.requireNonNull(image, "image"), patchSize, merge, preprocessing);
    }

    Lfm2VisionPreprocess.Plan plan(Media.Image image) {
        return Lfm2VisionPreprocess.plan(
                Objects.requireNonNull(image, "image"), patchSize, merge, preprocessing);
    }

    int positions(Lfm2VisionPreprocess.Part part) {
        return Lfm2VisionPreprocess.positions(part, patchSize, merge);
    }

    @Override
    public String planId() {
        return "lfm2v patch="
                + patchSize
                + " merge="
                + merge
                + " pos="
                + positionSide
                + " pixels="
                + preprocessing.minPixels()
                + ".."
                + preprocessing.maxPixels()
                + " tile="
                + preprocessing.tileSize()
                + " minTiles="
                + preprocessing.minTiles()
                + " maxTiles="
                + preprocessing.maxTiles()
                + " tolerance="
                + preprocessing.maxPixelsTolerance()
                + " mean="
                + Arrays.toString(imageMean)
                + " std="
                + Arrays.toString(imageStd);
    }

    @Override
    public void project(Media.Image image, int maxChunkSize, Consumer<MemoryView<?>> sink) {
        Objects.requireNonNull(sink, "sink");
        Views.checkAlive(patchWeight, "patchWeight"); // fail-fast on freed weights
        for (Lfm2VisionPreprocess.Part part : plan(image).parts()) embed(part, maxChunkSize, sink);
        Reference.reachabilityFence(this); // kernels read weights via raw bases; pin `this`
    }

    void embed(Lfm2VisionPreprocess.Part part, int maxChunkSize, Consumer<MemoryView<?>> sink) {
        Objects.requireNonNull(part, "part");
        Objects.requireNonNull(sink, "sink");
        if (maxChunkSize <= 0) throw new IllegalArgumentException("maxChunkSize must be positive");
        int rows = positions(part);
        MemoryArena<MemorySegment> scratch = Arenas.newCrossThreadMemoryArena();
        try {
            MemoryView<MemorySegment> encoded = encode(part.image(), scratch);
            for (int first = 0; first < rows; first += maxChunkSize)
                sink.accept(encoded.slice(0, first, Math.min(rows, first + maxChunkSize)));
        } finally {
            Arenas.close(scratch);
        }
    }

    private MemoryView<MemorySegment> encode(
            Media.Image image, MemoryArena<MemorySegment> scratch) {
        int patchesX = image.width() / patchSize, patchesY = image.height() / patchSize;
        int count = Math.multiplyExact(patchesX, patchesY);
        int patchVector = Math.multiplyExact(3, Math.multiplyExact(patchSize, patchSize));
        MemoryView<MemorySegment> patches =
                Lfm2VisionPreprocess.patches(image, patchSize, imageMean, imageStd, scratch);
        MemoryView<MemorySegment> current = Views.allocateF32(scratch, count, visionDim);
        MatMul.gemm(
                patchWeight,
                patches,
                patchVector,
                current,
                visionDim,
                visionDim,
                count,
                patchVector);
        Ops.addRowBiasInPlace(current, 0, patchBias, 0, count, visionDim);
        addInterpolatedPositions(current, patchesX, patchesY, scratch);

        MemoryView<MemorySegment> normalized = Views.allocateF32(scratch, count, visionDim);
        MemoryView<MemorySegment> query = Views.allocateF32(scratch, count, visionDim);
        MemoryView<MemorySegment> key = Views.allocateF32(scratch, count, visionDim);
        MemoryView<MemorySegment> value = Views.allocateF32(scratch, count, visionDim);
        MemoryView<MemorySegment> attention = Views.allocateF32(scratch, count, visionDim);
        MemoryView<MemorySegment> keyF16 = Views.allocateF16(scratch, count, visionDim);
        MemoryView<MemorySegment> valueF16 = Views.allocateF16(scratch, count, visionDim);
        MemoryView<MemorySegment> hidden = Views.allocateF32(scratch, count, ffnDim);
        float attentionScale = 1f / (float) Math.sqrt(headDim);

        for (Layer layer : layers) {
            Norms.layerNorm(
                    normalized,
                    current,
                    layer.norm1Weight(),
                    layer.norm1Bias(),
                    visionDim,
                    count,
                    normEps);
            linear(layer.query(), normalized, query, count);
            linear(layer.key(), normalized, key, count);
            linear(layer.value(), normalized, value, count);
            int elements = Math.multiplyExact(count, visionDim);
            Convert.f32ToF16(key, 0, keyF16, 0, elements);
            Convert.f32ToF16(value, 0, valueF16, 0, elements);
            FlashAttention.bidirectionalPrefill(
                    query,
                    attention,
                    keyF16,
                    valueF16,
                    headCount,
                    count,
                    headDim,
                    visionDim,
                    visionDim,
                    1,
                    attentionScale);
            linear(layer.attentionOutput(), attention, normalized, count);
            Ops.addInPlace(current, 0, normalized, 0, elements);

            Norms.layerNorm(
                    normalized,
                    current,
                    layer.norm2Weight(),
                    layer.norm2Bias(),
                    visionDim,
                    count,
                    normEps);
            linear(layer.ffnUp(), normalized, hidden, count);
            Activations.geluInPlace(hidden, 0, Math.multiplyExact(count, ffnDim));
            linear(layer.ffnDown(), hidden, normalized, count);
            Ops.addInPlace(current, 0, normalized, 0, elements);
        }
        Norms.layerNorm(
                normalized, current, postNormWeight, postNormBias, visionDim, count, normEps);
        return project(merge(normalized, patchesX, patchesY, merge, visionDim, scratch), scratch);
    }

    private void addInterpolatedPositions(
            MemoryView<MemorySegment> current,
            int targetWidth,
            int targetHeight,
            MemoryArena<MemorySegment> scratch) {
        float[] interpolated =
                interpolatePositions(
                        positionEmbedding, positionSide, visionDim, targetWidth, targetHeight);
        MemoryView<MemorySegment> positions =
                Views.allocateF32(
                        scratch, Math.multiplyExact(targetWidth, targetHeight), visionDim);
        Views.copyFromArray(positions, 0, interpolated, 0, interpolated.length, "positions");
        Ops.addInPlace(current, 0, positions, 0, interpolated.length);
    }

    static float[] interpolatePositions(
            float[] source, int sourceSide, int channels, int targetWidth, int targetHeight) {
        if (sourceSide <= 0 || channels <= 0 || targetWidth <= 0 || targetHeight <= 0)
            throw new IllegalArgumentException("invalid position geometry");
        if (source.length != (long) sourceSide * sourceSide * channels)
            throw new IllegalArgumentException("invalid position table size");
        if (targetWidth == sourceSide && targetHeight == sourceSide) return source.clone();
        float scaleX = (float) targetWidth / sourceSide;
        float scaleY = (float) targetHeight / sourceSide;
        float supportX = Math.max(1f, 1f / scaleX), inverseX = 1f / supportX;
        float supportY = Math.max(1f, 1f / scaleY), inverseY = 1f / supportY;
        float[] output =
                new float
                        [Math.multiplyExact(
                                Math.multiplyExact(targetWidth, targetHeight), channels)];
        Parallel.forLoop(
                targetHeight,
                y -> {
                    float sourceY = (y + 0.5f) / scaleY;
                    int yMin = Math.max((int) (sourceY - supportY + 0.5f), 0);
                    int yMax = Math.min((int) (sourceY + supportY + 0.5f), sourceSide);
                    for (int x = 0; x < targetWidth; x++) {
                        float sourceX = (x + 0.5f) / scaleX;
                        int xMin = Math.max((int) (sourceX - supportX + 0.5f), 0);
                        int xMax = Math.min((int) (sourceX + supportX + 0.5f), sourceSide);
                        float totalWeight = 0f;
                        for (int sy = yMin; sy < yMax; sy++) {
                            float wy =
                                    Math.max(1f - Math.abs((sy - sourceY + 0.5f) * inverseY), 0f);
                            for (int sx = xMin; sx < xMax; sx++) {
                                float wx =
                                        Math.max(
                                                1f - Math.abs((sx - sourceX + 0.5f) * inverseX),
                                                0f);
                                totalWeight += wx * wy;
                            }
                        }
                        int outputBase = (y * targetWidth + x) * channels;
                        for (int channel = 0; channel < channels; channel++) {
                            float sum = 0f;
                            for (int sy = yMin; sy < yMax; sy++) {
                                float wy =
                                        Math.max(
                                                1f - Math.abs((sy - sourceY + 0.5f) * inverseY),
                                                0f);
                                for (int sx = xMin; sx < xMax; sx++) {
                                    float wx =
                                            Math.max(
                                                    1f - Math.abs((sx - sourceX + 0.5f) * inverseX),
                                                    0f);
                                    sum +=
                                            source[(sy * sourceSide + sx) * channels + channel]
                                                    * wx
                                                    * wy;
                                }
                            }
                            output[outputBase + channel] = sum / totalWeight;
                        }
                    }
                });
        return output;
    }

    static MemoryView<MemorySegment> merge(
            MemoryView<MemorySegment> source,
            int patchesX,
            int patchesY,
            int merge,
            int visionDim,
            MemoryArena<MemorySegment> scratch) {
        if (patchesX % merge != 0 || patchesY % merge != 0)
            throw new IllegalArgumentException("patch grid must be divisible by merge");
        int outputX = patchesX / merge, outputY = patchesY / merge;
        int mergedDim = Math.multiplyExact(visionDim, Math.multiplyExact(merge, merge));
        MemoryView<MemorySegment> merged =
                Views.allocateF32(scratch, Math.multiplyExact(outputX, outputY), mergedDim);
        Parallel.forLoop(
                outputY,
                outputRow -> {
                    for (int outputColumn = 0; outputColumn < outputX; outputColumn++) {
                        int outputBase = (outputRow * outputX + outputColumn) * mergedDim;
                        for (int innerRow = 0; innerRow < merge; innerRow++)
                            for (int innerColumn = 0; innerColumn < merge; innerColumn++) {
                                int inputBase =
                                        ((outputRow * merge + innerRow) * patchesX
                                                        + outputColumn * merge
                                                        + innerColumn)
                                                * visionDim;
                                int featureBase =
                                        outputBase + (innerRow * merge + innerColumn) * visionDim;
                                Convert.copyF32(source, inputBase, merged, featureBase, visionDim);
                            }
                    }
                });
        return merged;
    }

    private MemoryView<MemorySegment> project(
            MemoryView<MemorySegment> merged, MemoryArena<MemorySegment> scratch) {
        int rows = Math.toIntExact(merged.shape().flatAt(0));
        int mergedDim = Math.multiplyExact(visionDim, Math.multiplyExact(merge, merge));
        if (projectorNormWeight != null)
            Norms.layerNorm(
                    merged, merged, projectorNormWeight, projectorNormBias, mergedDim, rows, 1e-5f);
        MemoryView<MemorySegment> hidden = Views.allocateF32(scratch, rows, projectorDim);
        linear(projectorUp, merged, hidden, rows);
        Activations.geluInPlace(hidden, 0, Math.multiplyExact(rows, projectorDim));
        MemoryView<MemorySegment> projected = Views.allocateF32(scratch, rows, modelDim);
        linear(projectorDown, hidden, projected, rows);
        return projected;
    }

    private static void linear(
            Linear linear,
            MemoryView<MemorySegment> input,
            MemoryView<MemorySegment> output,
            int rows) {
        MatMul.gemm(
                linear.weight(),
                input,
                linear.inputDim(),
                output,
                linear.outputDim(),
                linear.outputDim(),
                rows,
                linear.inputDim());
        Ops.addRowBiasInPlace(output, 0, linear.bias(), 0, rows, linear.outputDim());
    }

    private void validateLayer(Layer layer, int index) {
        Objects.requireNonNull(layer, "layer " + index);
        String prefix = "v.blk." + index + ".";
        requireF32(layer.norm1Weight(), prefix + "ln1.weight", Shape.flat(visionDim));
        requireF32(layer.norm1Bias(), prefix + "ln1.bias", Shape.flat(visionDim));
        requireLinear(layer.query(), prefix + "attn_q", visionDim, visionDim);
        requireLinear(layer.key(), prefix + "attn_k", visionDim, visionDim);
        requireLinear(layer.value(), prefix + "attn_v", visionDim, visionDim);
        requireLinear(layer.attentionOutput(), prefix + "attn_out", visionDim, visionDim);
        requireF32(layer.norm2Weight(), prefix + "ln2.weight", Shape.flat(visionDim));
        requireF32(layer.norm2Bias(), prefix + "ln2.bias", Shape.flat(visionDim));
        requireLinear(layer.ffnUp(), prefix + "ffn_up", ffnDim, visionDim);
        requireLinear(layer.ffnDown(), prefix + "ffn_down", visionDim, ffnDim);
    }

    public static Lfm2Vision loadModel(Path path, Arena arena) throws IOException {
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            GGUF gguf = ModelLoader.readGguf(channel, path.toString());
            return loadModel(path, gguf, ModelLoader.loadTensors(channel, gguf, arena));
        }
    }

    public static Lfm2Vision loadModel(
            Path label, GGUF gguf, Map<String, MemoryView<MemorySegment>> tensors) {
        Objects.requireNonNull(label, "label");
        Objects.requireNonNull(gguf, "gguf");
        Objects.requireNonNull(tensors, "tensors");
        String projectorType =
                gguf.getStringOrDefault(
                        "clip.vision.projector_type",
                        gguf.getStringOrDefault("clip.projector_type", ""));
        if (!"lfm2".equals(projectorType))
            throw new IllegalArgumentException(
                    label.getFileName()
                            + ": unsupported vision projector '"
                            + projectorType
                            + "' (expected lfm2)");
        int patchSize = gguf.getValueOrDefault(int.class, "clip.vision.patch_size", 16);
        int visionDim = gguf.getValueOrDefault(int.class, "clip.vision.embedding_length", 1152);
        int headCount = gguf.getValueOrDefault(int.class, "clip.vision.attention.head_count", 16);
        int layerCount = gguf.getValueOrDefault(int.class, "clip.vision.block_count", 27);
        int ffnDim = gguf.getValueOrDefault(int.class, "clip.vision.feed_forward_length", 4304);
        int merge =
                gguf.getValueOrDefault(
                        int.class,
                        "clip.vision.projector.scale_factor",
                        gguf.getValueOrDefault(int.class, "clip.vision.proj_scale_factor", 2));
        int modelDim = gguf.getValueOrDefault(int.class, "clip.vision.projection_dim", 2048);
        float normEps =
                gguf.getValueOrDefault(
                        float.class, "clip.vision.attention.layer_norm_epsilon", 1e-6f);
        if (patchSize <= 0
                || visionDim <= 0
                || headCount <= 0
                || layerCount <= 0
                || ffnDim <= 0
                || merge <= 1
                || modelDim <= 0)
            throw new IllegalArgumentException(label.getFileName() + ": invalid vision metadata");
        Lfm2VisionPreprocess.Options defaults = Lfm2VisionPreprocess.defaults(patchSize, merge);
        Lfm2VisionPreprocess.Options preprocessing =
                new Lfm2VisionPreprocess.Options(
                        gguf.getValueOrDefault(
                                int.class, "clip.vision.image_min_pixels", defaults.minPixels()),
                        gguf.getValueOrDefault(
                                int.class, "clip.vision.image_max_pixels", defaults.maxPixels()),
                        gguf.getValueOrDefault(
                                int.class, "clip.vision.preproc_image_size", defaults.tileSize()),
                        gguf.getValueOrDefault(
                                int.class, "clip.vision.preproc_min_tiles", defaults.minTiles()),
                        gguf.getValueOrDefault(
                                int.class, "clip.vision.preproc_max_tiles", defaults.maxTiles()),
                        defaults.maxPixelsTolerance());
        float[] imageMean =
                gguf.getValueOrDefault(
                        float[].class, "clip.vision.image_mean", new float[] {0.5f, 0.5f, 0.5f});
        float[] imageStd =
                gguf.getValueOrDefault(
                        float[].class, "clip.vision.image_std", new float[] {0.5f, 0.5f, 0.5f});

        MemoryView<MemorySegment> position = require(tensors, "v.position_embd.weight");
        requireF32(position, "v.position_embd.weight", position.shape());
        Shape positionShape = position.shape();
        if (!positionShape.isFlat()
                || positionShape.flatRank() != 2
                || positionShape.flatAt(1) != visionDim)
            throw new IllegalArgumentException(
                    label.getFileName()
                            + ": v.position_embd.weight expected [positions, "
                            + visionDim
                            + "] but was "
                            + positionShape);
        int positionCount = Math.toIntExact(positionShape.flatAt(0));
        int positionSide = (int) Math.sqrt(positionCount);
        if (positionSide * positionSide != positionCount)
            throw new IllegalArgumentException(
                    label.getFileName() + ": position count is not square: " + positionCount);

        Layer[] layers = new Layer[layerCount];
        for (int i = 0; i < layerCount; i++) {
            String prefix = "v.blk." + i + ".";
            layers[i] =
                    new Layer(
                            require(tensors, prefix + "ln1.weight"),
                            require(tensors, prefix + "ln1.bias"),
                            loadLinear(tensors, prefix + "attn_q", visionDim, visionDim),
                            loadLinear(tensors, prefix + "attn_k", visionDim, visionDim),
                            loadLinear(tensors, prefix + "attn_v", visionDim, visionDim),
                            loadLinear(tensors, prefix + "attn_out", visionDim, visionDim),
                            require(tensors, prefix + "ln2.weight"),
                            require(tensors, prefix + "ln2.bias"),
                            loadLinear(tensors, prefix + "ffn_up", ffnDim, visionDim),
                            loadLinear(tensors, prefix + "ffn_down", visionDim, ffnDim));
        }
        int projectorInput = Math.multiplyExact(visionDim, Math.multiplyExact(merge, merge));
        MemoryView<MemorySegment> projectorUpWeight = require(tensors, "mm.1.weight");
        Shape projectorUpShape =
                projectorUpWeight.dataType().logicalShape(projectorUpWeight.shape());
        if (!projectorUpShape.isFlat()
                || projectorUpShape.flatRank() != 2
                || projectorUpShape.flatAt(1) != projectorInput)
            throw new IllegalArgumentException(
                    label.getFileName()
                            + ": mm.1.weight expected input width "
                            + projectorInput
                            + " but was "
                            + projectorUpShape);
        int projectorDim = Math.toIntExact(projectorUpShape.flatAt(0));
        return new Lfm2Vision(
                patchSize,
                visionDim,
                headCount,
                ffnDim,
                merge,
                projectorDim,
                modelDim,
                positionSide,
                normEps,
                preprocessing,
                imageMean,
                imageStd,
                require(tensors, "v.patch_embd.weight"),
                require(tensors, "v.patch_embd.bias"),
                Views.toFloatArray(position, "v.position_embd.weight"),
                require(tensors, "v.post_ln.weight"),
                require(tensors, "v.post_ln.bias"),
                tensors.get("mm.input_norm.weight"),
                tensors.get("mm.input_norm.bias"),
                loadLinear(tensors, "mm.1", projectorDim, projectorInput),
                loadLinear(tensors, "mm.2", modelDim, projectorDim),
                layers);
    }

    private static float[] requireNormalization(float[] values, String name, boolean positive) {
        Objects.requireNonNull(values, name);
        if (values.length != 3)
            throw new IllegalArgumentException(name + " must contain three channels");
        float[] copy = values.clone();
        for (float value : copy)
            if (!Float.isFinite(value) || (positive && !(value > 0f)))
                throw new IllegalArgumentException(name + " contains an invalid value");
        return copy;
    }

    private static Linear loadLinear(
            Map<String, MemoryView<MemorySegment>> tensors,
            String name,
            int outputDim,
            int inputDim) {
        return new Linear(
                require(tensors, name + ".weight"),
                require(tensors, name + ".bias"),
                outputDim,
                inputDim);
    }

    private static Linear requireLinear(Linear linear, String name, int outputDim, int inputDim) {
        Objects.requireNonNull(linear, name);
        if (linear.outputDim() != outputDim || linear.inputDim() != inputDim)
            throw new IllegalArgumentException(name + ": invalid dimensions");
        requireWeight(linear.weight(), name + ".weight", Shape.flat(outputDim, inputDim));
        requireF32(linear.bias(), name + ".bias", Shape.flat(outputDim));
        return linear;
    }

    private static MemoryView<MemorySegment> requireWeight(
            MemoryView<MemorySegment> value, String name, Shape expected) {
        Objects.requireNonNull(value, name);
        Views.requireContiguous(value, name);
        DataType type = value.dataType();
        if (type != DataType.FP32
                && type != DataType.FP16
                && type != DataType.BF16
                && type != DataType.Q8_0)
            throw new IllegalArgumentException(name + ": unsupported weight type " + type.name());
        Shape actual = type.logicalShape(value.shape());
        if (!actual.equals(expected))
            throw new IllegalArgumentException(
                    name + ": expected shape " + expected + " but was " + actual);
        return value;
    }

    private static MemoryView<MemorySegment> requirePatchWeight(
            MemoryView<MemorySegment> value, String name, int outputDim, int inputDim) {
        Objects.requireNonNull(value, name);
        Views.requireContiguous(value, name);
        DataType type = value.dataType();
        if (type != DataType.FP32
                && type != DataType.FP16
                && type != DataType.BF16
                && type != DataType.Q8_0)
            throw new IllegalArgumentException(name + ": unsupported weight type " + type.name());
        Shape actual = type.logicalShape(value.shape());
        if (!actual.isFlat()
                || actual.flatAt(0) != outputDim
                || actual.size() != (long) outputDim * inputDim)
            throw new IllegalArgumentException(
                    name
                            + ": expected output/input "
                            + outputDim
                            + "x"
                            + inputDim
                            + " but was "
                            + actual);
        return value;
    }

    private static MemoryView<MemorySegment> requireF32(
            MemoryView<MemorySegment> value, String name, Shape expected) {
        Objects.requireNonNull(value, name);
        Views.requireDense(value, DataType.FP32, name);
        if (!value.shape().equals(expected))
            throw new IllegalArgumentException(
                    name + ": expected shape " + expected + " but was " + value.shape());
        return value;
    }
}
