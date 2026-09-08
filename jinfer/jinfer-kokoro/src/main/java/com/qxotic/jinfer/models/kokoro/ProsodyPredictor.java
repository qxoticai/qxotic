package com.qxotic.jinfer.models.kokoro;

import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.kernels.Convert;
import com.qxotic.jinfer.kernels.ModelLoader;
import com.qxotic.jinfer.kernels.Ops;
import com.qxotic.jota.DataType;
import com.qxotic.jota.memory.MemoryAllocator;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/** Kokoro's duration, pitch, and noise predictor. */
final class ProsodyPredictor {

    private static final float LEAKY_RELU_SLOPE = 0.2f;
    private static final float RESIDUAL_SCALE = (float) (1.0 / Math.sqrt(2.0));

    record DurationLayer(
            KokoroOps.LstmWeights forward,
            KokoroOps.LstmWeights reverse,
            KokoroLayers.AdaLayerNorm normalization) {}

    record AdainResBlk1d(
            KokoroLayers.AdaIN normalization1,
            KokoroLayers.AdaIN normalization2,
            KokoroLayers.DepthwiseUpsample pool,
            KokoroLayers.Conv1d convolution1,
            KokoroLayers.Conv1d convolution2,
            KokoroLayers.Conv1d shortcut) {

        MemoryView<MemorySegment> forward(
                MemoryView<MemorySegment> input,
                int time,
                MemoryView<MemorySegment> style,
                MemoryAllocator<MemorySegment> scratch) {
            int outputTime = pool == null ? time : Math.multiplyExact(time, 2);
            MemoryView<MemorySegment> residual =
                    normalization1.forward(input, time, style, scratch);
            Ops.leakyReluInPlace(
                    residual, 0, Math.toIntExact(residual.logicalSize()), LEAKY_RELU_SLOPE);
            if (pool != null) residual = pool.forward(residual, time, scratch);
            residual = convolution1.forward(residual, outputTime, scratch);
            residual = normalization2.forward(residual, outputTime, style, scratch);
            Ops.leakyReluInPlace(
                    residual, 0, Math.toIntExact(residual.logicalSize()), LEAKY_RELU_SLOPE);
            residual = convolution2.forward(residual, outputTime, scratch);

            MemoryView<MemorySegment> direct = input;
            if (pool != null)
                direct =
                        KokoroLayers.nearestUpsample(
                                direct, normalization1.channels(), time, 2, scratch);
            if (shortcut != null) direct = shortcut.forward(direct, outputTime, scratch);
            int elements = Math.toIntExact(residual.logicalSize());
            require(direct.logicalSize() == elements, "residual branches have different sizes");
            Ops.addInPlace(residual, 0, direct, 0, elements);
            Ops.multiplyInPlace(residual, 0, elements, RESIDUAL_SCALE);
            return residual;
        }
    }

    record Branch(List<AdainResBlk1d> blocks, KokoroLayers.Conv1d projection) {
        Branch {
            blocks = List.copyOf(blocks);
        }
    }

    record Weights(
            List<DurationLayer> durationEncoder,
            KokoroOps.LstmWeights durationForward,
            KokoroOps.LstmWeights durationReverse,
            KokoroLayers.Linear durationProjection,
            KokoroOps.LstmWeights sharedForward,
            KokoroOps.LstmWeights sharedReverse,
            Branch f0,
            Branch noise,
            int hidden,
            int style,
            int maxDuration) {
        Weights {
            durationEncoder = List.copyOf(durationEncoder);
        }
    }

    record Output(int[] alignmentIndices, float[] f0, float[] noise) {}

    private ProsodyPredictor() {}

    static Weights load(
            Map<String, MemoryView<MemorySegment>> tensors,
            Kokoro.Configuration config,
            MemoryAllocator<MemorySegment> persistent) {
        int hidden = config.hiddenDim();
        int style = config.styleDim();
        int encoded = Math.addExact(hidden, style);

        List<DurationLayer> durationEncoder = new ArrayList<>(3);
        for (int layer = 0; layer < 3; layer++) {
            String prefix = "pred.dur_enc." + layer;
            durationEncoder.add(
                    new DurationLayer(
                            lstm(tensors, persistent, prefix + ".lstm", "", encoded, hidden / 2),
                            lstm(
                                    tensors,
                                    persistent,
                                    prefix + ".lstm",
                                    "_reverse",
                                    encoded,
                                    hidden / 2),
                            new KokoroLayers.AdaLayerNorm(
                                    KokoroLayers.linear(
                                            tensors,
                                            persistent,
                                            prefix + ".adaln",
                                            style,
                                            2 * hidden),
                                    hidden)));
        }

        return new Weights(
                durationEncoder,
                lstm(tensors, persistent, "pred.lstm", "", encoded, hidden / 2),
                lstm(tensors, persistent, "pred.lstm", "_reverse", encoded, hidden / 2),
                KokoroLayers.linear(
                        tensors, persistent, "pred.dur_proj", hidden, config.maxDuration()),
                lstm(tensors, persistent, "pred.shared", "", encoded, hidden / 2),
                lstm(tensors, persistent, "pred.shared", "_reverse", encoded, hidden / 2),
                branch(tensors, persistent, "pred.F0", "pred.F0_proj", hidden, style),
                branch(tensors, persistent, "pred.N", "pred.N_proj", hidden, style),
                hidden,
                style,
                config.maxDuration());
    }

    static Output forward(
            Weights weights,
            MemoryView<MemorySegment> bertProjected,
            MemoryView<MemorySegment> predictorStyle,
            int length,
            double speed,
            MemoryAllocator<MemorySegment> scratch) {
        require(length > 0, "predictor input is empty");
        checkMatrix(bertProjected, length, weights.hidden(), "BERT projection");
        checkMatrix(predictorStyle, 1, weights.style(), "predictor style");

        MemoryView<MemorySegment> durationEncoding =
                appendStyle(
                        bertProjected,
                        predictorStyle,
                        length,
                        weights.hidden(),
                        weights.style(),
                        scratch);
        MemoryView<MemorySegment> nextDuration =
                Views.allocateF32(scratch, length, weights.hidden() + weights.style());
        for (DurationLayer layer : weights.durationEncoder()) {
            try (var ignored = KokoroWorkspace.scope(scratch)) {
                MemoryView<MemorySegment> recurrent =
                        Views.allocateF32(scratch, length, weights.hidden());
                KokoroOps.bidirectionalLstm(
                        durationEncoding, layer.forward(), layer.reverse(), recurrent, scratch);
                MemoryView<MemorySegment> normalized =
                        layer.normalization().forward(recurrent, length, predictorStyle, scratch);
                appendStyle(
                        nextDuration,
                        normalized,
                        predictorStyle,
                        length,
                        weights.hidden(),
                        weights.style());
            }
            MemoryView<MemorySegment> swap = durationEncoding;
            durationEncoding = nextDuration;
            nextDuration = swap;
        }

        MemoryView<MemorySegment> durationState =
                Views.allocateF32(scratch, length, weights.hidden());
        KokoroOps.bidirectionalLstm(
                durationEncoding,
                weights.durationForward(),
                weights.durationReverse(),
                durationState,
                scratch);
        MemoryView<MemorySegment> logits =
                weights.durationProjection().forward(durationState, length, scratch);
        int[] durations =
                KokoroOps.durations(logits, length, weights.maxDuration(), speed, scratch);
        int[] alignment = repeatAlignment(durations, scratch);

        MemoryView<MemorySegment> gathered =
                gatherRows(
                        durationEncoding,
                        alignment,
                        Math.addExact(weights.hidden(), weights.style()),
                        scratch);
        int frames = alignment.length;
        MemoryView<MemorySegment> shared = Views.allocateF32(scratch, frames, weights.hidden());
        KokoroOps.bidirectionalLstm(
                gathered, weights.sharedForward(), weights.sharedReverse(), shared, scratch);
        MemoryView<MemorySegment> channelMajor =
                Views.allocateF32(scratch, weights.hidden(), frames);
        Ops.transposeCopy(shared, frames, weights.hidden(), channelMajor);

        float[] f0 = KokoroWorkspace.takeFloats(scratch, 2 * frames);
        try (var ignored = KokoroWorkspace.scope(scratch)) {
            runBranch(weights.f0(), channelMajor, frames, predictorStyle, scratch, f0);
        }
        float[] noise = KokoroWorkspace.takeFloats(scratch, 2 * frames);
        try (var ignored = KokoroWorkspace.scope(scratch)) {
            runBranch(weights.noise(), channelMajor, frames, predictorStyle, scratch, noise);
        }
        return new Output(alignment, f0, noise);
    }

    static int[] repeatAlignment(int[] durations, MemoryAllocator<MemorySegment> scratch) {
        int size = 0;
        for (int duration : durations) {
            require(duration > 0, "durations must be positive");
            size = Math.addExact(size, duration);
        }
        int[] indices = KokoroWorkspace.takeInts(scratch, size);
        int at = 0;
        for (int token = 0; token < durations.length; token++)
            for (int frame = 0; frame < durations[token]; frame++) indices[at++] = token;
        return indices;
    }

    static MemoryView<MemorySegment> appendStyle(
            MemoryView<MemorySegment> input,
            MemoryView<MemorySegment> style,
            int rows,
            int channels,
            int styleChannels,
            MemoryAllocator<MemorySegment> allocator) {
        checkMatrix(input, rows, channels, "style concatenation input");
        checkMatrix(style, 1, styleChannels, "style concatenation style");
        int width = Math.addExact(channels, styleChannels);
        MemoryView<MemorySegment> result = Views.allocateF32(allocator, rows, width);
        appendStyle(result, input, style, rows, channels, styleChannels);
        return result;
    }

    private static void appendStyle(
            MemoryView<MemorySegment> result,
            MemoryView<MemorySegment> input,
            MemoryView<MemorySegment> style,
            int rows,
            int channels,
            int styleChannels) {
        int width = Math.addExact(channels, styleChannels);
        for (int row = 0; row < rows; row++) {
            Convert.copyF32(input, (long) row * channels, result, (long) row * width, channels);
            Convert.copyF32(style, 0, result, (long) row * width + channels, styleChannels);
        }
    }

    static MemoryView<MemorySegment> gatherRows(
            MemoryView<MemorySegment> input,
            int[] indices,
            int width,
            MemoryAllocator<MemorySegment> allocator) {
        require(input.shape().flatRank() == 2, "gather input must be a matrix");
        require(input.shape().flatAt(1) == width, "gather input has the wrong width");
        int rows = Math.toIntExact(input.shape().flatAt(0));
        for (int index : indices) require(index >= 0 && index < rows, "invalid alignment index");
        MemoryView<MemorySegment> result = Views.allocateF32(allocator, indices.length, width);
        Convert.gatherToF32(input, indices, 0, indices.length, result, 0, width);
        return result;
    }

    private static Branch branch(
            Map<String, MemoryView<MemorySegment>> tensors,
            MemoryAllocator<MemorySegment> allocator,
            String prefix,
            String projection,
            int hidden,
            int style) {
        return new Branch(
                List.of(
                        block(tensors, allocator, prefix + ".0", hidden, hidden, style, false),
                        block(tensors, allocator, prefix + ".1", hidden, hidden / 2, style, true),
                        block(
                                tensors,
                                allocator,
                                prefix + ".2",
                                hidden / 2,
                                hidden / 2,
                                style,
                                false)),
                KokoroLayers.conv1d(tensors, allocator, projection, 1, hidden / 2, 1));
    }

    private static AdainResBlk1d block(
            Map<String, MemoryView<MemorySegment>> tensors,
            MemoryAllocator<MemorySegment> allocator,
            String prefix,
            int inputChannels,
            int outputChannels,
            int style,
            boolean upsample) {
        return new AdainResBlk1d(
                new KokoroLayers.AdaIN(
                        KokoroLayers.linear(
                                tensors, allocator, prefix + ".adain1", style, 2 * inputChannels),
                        inputChannels),
                new KokoroLayers.AdaIN(
                        KokoroLayers.linear(
                                tensors, allocator, prefix + ".adain2", style, 2 * outputChannels),
                        outputChannels),
                upsample
                        ? KokoroLayers.depthwiseUpsample(
                                tensors, allocator, prefix + ".pool", inputChannels)
                        : null,
                KokoroLayers.conv1d(
                        tensors, allocator, prefix + ".conv1", 3, inputChannels, outputChannels),
                KokoroLayers.conv1d(
                        tensors, allocator, prefix + ".conv2", 3, outputChannels, outputChannels),
                inputChannels == outputChannels
                        ? null
                        : KokoroLayers.conv1d(
                                tensors,
                                allocator,
                                prefix + ".conv1x1",
                                1,
                                inputChannels,
                                outputChannels));
    }

    private static void runBranch(
            Branch branch,
            MemoryView<MemorySegment> input,
            int frames,
            MemoryView<MemorySegment> style,
            MemoryAllocator<MemorySegment> scratch,
            float[] result) {
        MemoryView<MemorySegment> current = input;
        int time = frames;
        for (AdainResBlk1d block : branch.blocks()) {
            current = block.forward(current, time, style, scratch);
            if (block.pool() != null) time = Math.multiplyExact(time, 2);
        }
        MemoryView<MemorySegment> projected = branch.projection().forward(current, time, scratch);
        require(result.length == time, "prediction output has the wrong length");
        Views.copyToArray(projected, 0, result, 0, time, "prediction");
    }

    private static KokoroOps.LstmWeights lstm(
            Map<String, MemoryView<MemorySegment>> tensors,
            MemoryAllocator<MemorySegment> allocator,
            String prefix,
            String suffix,
            int input,
            int hidden) {
        int gates = Math.multiplyExact(4, hidden);
        return new KokoroOps.LstmWeights(
                matrix(tensors, prefix + ".weight_ih_l0" + suffix, gates, input),
                matrix(tensors, prefix + ".weight_hh_l0" + suffix, gates, hidden),
                KokoroLayers.vector(tensors, allocator, prefix + ".bias_ih_l0" + suffix, gates),
                KokoroLayers.vector(tensors, allocator, prefix + ".bias_hh_l0" + suffix, gates));
    }

    private static MemoryView<MemorySegment> matrix(
            Map<String, MemoryView<MemorySegment>> tensors, String name, int rows, int columns) {
        MemoryView<MemorySegment> value = ModelLoader.require(tensors, name);
        Views.requireContiguous(value, name);
        require(value.shape().flatRank() == 2, name + " must be a matrix");
        require(
                value.shape().flatAt(0) == rows
                        && value.shape().flatAt(1) * value.dataType().elementsPerBlock() == columns,
                name + " has the wrong shape");
        return value;
    }

    private static void checkMatrix(
            MemoryView<MemorySegment> view, int rows, int columns, String name) {
        Views.requireDense(view, DataType.FP32, name);
        require(
                view.shape().flatRank() == 2
                        && view.shape().flatAt(0) == rows
                        && view.shape().flatAt(1) == columns,
                name + " must be [" + rows + ", " + columns + "]");
    }

    private static void require(boolean condition, String message) {
        if (!condition) throw new IllegalArgumentException("Kokoro: " + message);
    }
}
