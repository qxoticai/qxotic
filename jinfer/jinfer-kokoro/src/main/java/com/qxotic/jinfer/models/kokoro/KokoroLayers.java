package com.qxotic.jinfer.models.kokoro;

import static com.qxotic.jinfer.Segments.readFloat;
import static com.qxotic.jinfer.Segments.writeFloat;

import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.kernels.Convert;
import com.qxotic.jinfer.kernels.Convolutions;
import com.qxotic.jinfer.kernels.MatMul;
import com.qxotic.jinfer.kernels.ModelLoader;
import com.qxotic.jinfer.kernels.Ops;
import com.qxotic.jota.DataType;
import com.qxotic.jota.memory.MemoryAllocator;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.Map;

/** Small reusable layers shared by Kokoro's predictor and decoder. */
final class KokoroLayers {

    private static final float ADAPTIVE_NORM_EPS = 1e-5f;

    record Linear(
            MemoryView<MemorySegment> weight,
            MemoryView<MemorySegment> bias,
            int inputSize,
            int outputSize) {

        MemoryView<MemorySegment> forward(
                MemoryView<MemorySegment> input,
                int rows,
                MemoryAllocator<MemorySegment> allocator) {
            MemoryView<MemorySegment> output = Views.allocateF32(allocator, rows, outputSize);
            MatMul.gemm(weight, input, inputSize, output, outputSize, outputSize, rows, inputSize);
            Ops.addRowBiasInPlace(output, 0, bias, 0, rows, outputSize);
            return output;
        }
    }

    /** Taps are laid out {@code [outChannel][inChannel][kernel]}. */
    record Conv1d(
            float[] taps,
            MemoryView<MemorySegment> bias,
            int kernel,
            int inChannels,
            int outChannels) {

        MemoryView<MemorySegment> forward(
                MemoryView<MemorySegment> input,
                int time,
                MemoryAllocator<MemorySegment> allocator) {
            checkChannelMajor(input, inChannels, time, "conv1d input");
            MemoryView<MemorySegment> output = Views.allocateF32(allocator, outChannels, time);
            Convolutions.conv1dRows(
                    input, inChannels, output, outChannels, time, kernel, 1, taps, bias);
            return output;
        }

        MemoryView<MemorySegment> forward(
                MemoryView<MemorySegment> input,
                int time,
                int stride,
                int padding,
                MemoryAllocator<MemorySegment> allocator) {
            require(stride > 0 && padding >= 0, "invalid convolution stride or padding");
            checkChannelMajor(input, inChannels, time, "conv1d input");
            int outTime = (time + 2 * padding - kernel) / stride + 1;
            require(outTime >= 0, "convolution output length is negative");
            MemoryView<MemorySegment> output = Views.allocateF32(allocator, outChannels, outTime);
            for (int oc = 0; oc < outChannels; oc++) {
                float b = bias == null ? 0f : Views.getFloat(bias, oc, "conv1d bias");
                for (int t = 0; t < outTime; t++) {
                    float sum = b;
                    for (int ic = 0; ic < inChannels; ic++) {
                        int tap = (oc * inChannels + ic) * kernel;
                        for (int k = 0; k < kernel; k++) {
                            int source = t * stride + k - padding;
                            if (source >= 0 && source < time)
                                sum += taps[tap + k] * get(input, (long) ic * time + source);
                        }
                    }
                    set(output, (long) oc * outTime + t, sum);
                }
            }
            return output;
        }
    }

    /** Taps are laid out {@code [outChannel][kernel][inChannel]}. */
    record ConvTranspose1d(
            float[] taps,
            MemoryView<MemorySegment> bias,
            int kernel,
            int inChannels,
            int outChannels) {

        MemoryView<MemorySegment> forward(
                MemoryView<MemorySegment> input,
                int time,
                int stride,
                int padding,
                int outputPadding,
                MemoryAllocator<MemorySegment> allocator) {
            require(
                    stride > 0 && padding >= 0 && outputPadding >= 0 && outputPadding < stride,
                    "invalid transposed convolution parameters");
            checkChannelMajor(input, inChannels, time, "conv transpose input");
            int outTime =
                    Math.addExact(
                            Math.subtractExact(
                                    Math.addExact(Math.multiplyExact(time - 1, stride), kernel),
                                    Math.multiplyExact(2, padding)),
                            outputPadding);
            require(outTime >= 0, "transposed convolution output length is negative");
            MemoryView<MemorySegment> output = Views.allocateF32(allocator, outChannels, outTime);
            float[] source = KokoroWorkspace.takeFloats(allocator, inChannels * time);
            float[] result = KokoroWorkspace.takeFloats(allocator, outChannels * outTime);
            Views.copyToArray(input, 0, source, 0, source.length, "conv transpose input");
            for (int oc = 0; oc < outChannels; oc++) {
                float b = bias == null ? 0f : Views.getFloat(bias, oc, "conv transpose bias");
                int outputRow = oc * outTime;
                Arrays.fill(result, outputRow, outputRow + outTime, b);
                int tapRow = oc * kernel * inChannels;
                for (int k = 0; k < kernel; k++) {
                    int tap = tapRow + k * inChannels;
                    for (int ic = 0; ic < inChannels; ic++) {
                        float weight = taps[tap + ic];
                        int inputIndex = ic * time;
                        int target = k - padding;
                        for (int t = 0; t < time; t++, target += stride) {
                            if (target >= 0 && target < outTime)
                                result[outputRow + target] += source[inputIndex + t] * weight;
                        }
                    }
                }
            }
            Views.copyFromArray(output, 0, result, 0, result.length, "conv transpose output");
            return output;
        }
    }

    /** Kokoro's grouped ConvTranspose1d(C,C,3,2,1,outputPadding=1). */
    record DepthwiseUpsample(float[] taps, MemoryView<MemorySegment> bias, int channels) {

        MemoryView<MemorySegment> forward(
                MemoryView<MemorySegment> input,
                int time,
                MemoryAllocator<MemorySegment> allocator) {
            checkChannelMajor(input, channels, time, "depthwise upsample input");
            int outTime = 2 * time;
            MemoryView<MemorySegment> output = Views.allocateF32(allocator, channels, outTime);
            for (int c = 0; c < channels; c++) {
                float b = bias == null ? 0f : Views.getFloat(bias, c, "depthwise upsample bias");
                for (int t = 0; t < time; t++) {
                    float value = get(input, (long) c * time + t);
                    set(output, (long) c * outTime + 2 * t, b + value * taps[c * 3 + 1]);
                    float odd = value * taps[c * 3 + 2];
                    if (t + 1 < time) odd += get(input, (long) c * time + t + 1) * taps[c * 3];
                    set(output, (long) c * outTime + 2 * t + 1, b + odd);
                }
            }
            return output;
        }
    }

    record AdaLayerNorm(Linear style, int channels) {

        MemoryView<MemorySegment> forward(
                MemoryView<MemorySegment> input,
                int steps,
                MemoryView<MemorySegment> styleVector,
                MemoryAllocator<MemorySegment> allocator) {
            checkTimeMajor(input, steps, channels, "AdaLayerNorm input");
            require(
                    style.outputSize == 2 * channels,
                    "AdaLayerNorm style projection is too narrow");
            MemoryView<MemorySegment> affine = style.forward(styleVector, 1, allocator);
            MemoryView<MemorySegment> output = Views.allocateF32(allocator, steps, channels);
            for (int t = 0; t < steps; t++) {
                int row = t * channels;
                double mean = 0;
                for (int c = 0; c < channels; c++) mean += get(input, row + c);
                mean /= channels;
                double variance = 0;
                for (int c = 0; c < channels; c++) {
                    double centered = get(input, row + c) - mean;
                    variance += centered * centered;
                }
                float scale = (float) (1.0 / Math.sqrt(variance / channels + ADAPTIVE_NORM_EPS));
                for (int c = 0; c < channels; c++)
                    set(
                            output,
                            row + c,
                            (get(input, row + c) - (float) mean) * scale * (1f + get(affine, c))
                                    + get(affine, channels + c));
            }
            return output;
        }
    }

    record AdaIN(Linear style, int channels) {

        MemoryView<MemorySegment> forward(
                MemoryView<MemorySegment> input,
                int time,
                MemoryView<MemorySegment> styleVector,
                MemoryAllocator<MemorySegment> allocator) {
            checkChannelMajor(input, channels, time, "AdaIN input");
            require(style.outputSize == 2 * channels, "AdaIN style projection is too narrow");
            MemoryView<MemorySegment> affine = style.forward(styleVector, 1, allocator);
            MemoryView<MemorySegment> output = Views.allocateF32(allocator, channels, time);
            for (int c = 0; c < channels; c++) {
                int row = c * time;
                double mean = 0;
                for (int t = 0; t < time; t++) mean += get(input, row + t);
                mean /= time;
                double variance = 0;
                for (int t = 0; t < time; t++) {
                    double centered = get(input, row + t) - mean;
                    variance += centered * centered;
                }
                float scale = (float) (1.0 / Math.sqrt(variance / time + ADAPTIVE_NORM_EPS));
                for (int t = 0; t < time; t++)
                    set(
                            output,
                            row + t,
                            (get(input, row + t) - (float) mean) * scale * (1f + get(affine, c))
                                    + get(affine, channels + c));
            }
            return output;
        }
    }

    record Snake(MemoryView<MemorySegment> alpha, int channels) {

        MemoryView<MemorySegment> forward(
                MemoryView<MemorySegment> input,
                int time,
                MemoryAllocator<MemorySegment> allocator) {
            checkChannelMajor(input, channels, time, "Snake input");
            MemoryView<MemorySegment> output = Views.allocateF32(allocator, channels, time);
            for (int c = 0; c < channels; c++) {
                float a = Views.getFloat(alpha, c, "Snake alpha");
                require(a != 0f, "Snake alpha must be non-zero");
                for (int t = 0; t < time; t++) {
                    int i = c * time + t;
                    float value = get(input, i);
                    double sine = Math.sin(a * value);
                    set(output, i, value + (float) (sine * sine / a));
                }
            }
            return output;
        }
    }

    private KokoroLayers() {}

    static Linear linear(
            Map<String, MemoryView<MemorySegment>> tensors,
            MemoryAllocator<MemorySegment> allocator,
            String name,
            int inputSize,
            int outputSize) {
        MemoryView<MemorySegment> weight = ModelLoader.require(tensors, name + ".weight");
        Views.requireContiguous(weight, name + ".weight");
        require(weight.shape().flatRank() == 2, name + ".weight must be a matrix");
        require(
                weight.shape().flatAt(0) == outputSize
                        && weight.shape().flatAt(1) * weight.dataType().elementsPerBlock()
                                == inputSize,
                name + ".weight has the wrong shape");
        return new Linear(
                weight,
                vector(tensors, allocator, name + ".bias", outputSize),
                inputSize,
                outputSize);
    }

    static Conv1d conv1d(
            Map<String, MemoryView<MemorySegment>> tensors,
            MemoryAllocator<MemorySegment> allocator,
            String name,
            int kernel,
            int inChannels,
            int outChannels) {
        return new Conv1d(
                convolutionTaps(
                        tensors, allocator, name + ".weight", kernel, inChannels, outChannels),
                optionalVector(tensors, allocator, name + ".bias", outChannels),
                kernel,
                inChannels,
                outChannels);
    }

    static ConvTranspose1d convTranspose1d(
            Map<String, MemoryView<MemorySegment>> tensors,
            MemoryAllocator<MemorySegment> allocator,
            String name,
            int kernel,
            int inChannels,
            int outChannels) {
        MemoryView<MemorySegment> weight = ModelLoader.require(tensors, name + ".weight");
        int taps = Math.multiplyExact(kernel, inChannels);
        require(weight.logicalSize() % outChannels == 0, name + ".weight has invalid rows");
        float[] result = new float[Math.multiplyExact(outChannels, taps)];
        if (weight.dataType().elementsPerBlock() > 1) {
            int rowStride = Math.toIntExact(weight.logicalSize() / outChannels);
            require(rowStride >= taps, name + ".weight rows are too short");
            MemoryView<MemorySegment> row = Views.allocateF32(allocator, taps);
            for (int oc = 0; oc < outChannels; oc++) {
                Convert.copyToF32(weight, (long) oc * rowStride, row, 0, taps);
                Views.copyToArray(row, 0, result, oc * taps, taps, name + ".weight");
            }
        } else {
            require(
                    weight.logicalSize() == (long) inChannels * outChannels * kernel,
                    name + ".weight has the wrong size");
            MemoryView<MemorySegment> dense = dequantize(allocator, weight, name + ".weight");
            float[] stored = Views.toFloatArray(dense, name + ".weight");
            for (int oc = 0; oc < outChannels; oc++)
                for (int k = 0; k < kernel; k++)
                    for (int ic = 0; ic < inChannels; ic++)
                        result[(oc * kernel + k) * inChannels + ic] =
                                stored[(ic * outChannels + oc) * kernel + k];
        }
        return new ConvTranspose1d(
                result,
                optionalVector(tensors, allocator, name + ".bias", outChannels),
                kernel,
                inChannels,
                outChannels);
    }

    static DepthwiseUpsample depthwiseUpsample(
            Map<String, MemoryView<MemorySegment>> tensors,
            MemoryAllocator<MemorySegment> allocator,
            String name,
            int channels) {
        return new DepthwiseUpsample(
                convolutionTaps(tensors, allocator, name + ".weight", 3, 1, channels),
                optionalVector(tensors, allocator, name + ".bias", channels),
                channels);
    }

    static MemoryView<MemorySegment> nearestUpsample(
            MemoryView<MemorySegment> input,
            int channels,
            int time,
            int scale,
            MemoryAllocator<MemorySegment> allocator) {
        require(scale > 0, "nearest upsample scale must be positive");
        checkChannelMajor(input, channels, time, "nearest upsample input");
        int outTime = Math.multiplyExact(time, scale);
        MemoryView<MemorySegment> output = Views.allocateF32(allocator, channels, outTime);
        for (int c = 0; c < channels; c++)
            for (int t = 0; t < outTime; t++)
                set(output, (long) c * outTime + t, get(input, (long) c * time + t / scale));
        return output;
    }

    static MemoryView<MemorySegment> vector(
            Map<String, MemoryView<MemorySegment>> tensors,
            MemoryAllocator<MemorySegment> allocator,
            String name,
            int size) {
        MemoryView<MemorySegment> source = ModelLoader.require(tensors, name);
        require(source.logicalSize() == size, name + " has the wrong length");
        return dequantize(allocator, source, name);
    }

    static float[] convolutionTaps(
            Map<String, MemoryView<MemorySegment>> tensors,
            MemoryAllocator<MemorySegment> allocator,
            String name,
            int kernel,
            int inChannels,
            int outChannels) {
        MemoryView<MemorySegment> weight = ModelLoader.require(tensors, name);
        int taps = Math.multiplyExact(kernel, inChannels);
        require(weight.logicalSize() % outChannels == 0, name + " has invalid rows");
        int rowStride = Math.toIntExact(weight.logicalSize() / outChannels);
        require(rowStride >= taps, name + " rows are too short");
        MemoryView<MemorySegment> row = Views.allocateF32(allocator, taps);
        float[] result = new float[Math.multiplyExact(outChannels, taps)];
        for (int oc = 0; oc < outChannels; oc++) {
            Convert.copyToF32(weight, (long) oc * rowStride, row, 0, taps);
            Views.copyToArray(row, 0, result, oc * taps, taps, name);
        }
        return result;
    }

    private static MemoryView<MemorySegment> optionalVector(
            Map<String, MemoryView<MemorySegment>> tensors,
            MemoryAllocator<MemorySegment> allocator,
            String name,
            int size) {
        return tensors.containsKey(name) ? vector(tensors, allocator, name, size) : null;
    }

    private static MemoryView<MemorySegment> dequantize(
            MemoryAllocator<MemorySegment> allocator,
            MemoryView<MemorySegment> source,
            String name) {
        if (source.dataType() == DataType.FP32) return source;
        int size = Math.toIntExact(source.logicalSize());
        MemoryView<MemorySegment> result = Views.allocateF32(allocator, size);
        Convert.copyToF32(source, 0, result, 0, size);
        return result;
    }

    private static float get(MemoryView<MemorySegment> view, long index) {
        return readFloat(view.memory().base(), view.byteOffset() + index * Float.BYTES);
    }

    private static void set(MemoryView<MemorySegment> view, long index, float value) {
        writeFloat(view.memory().base(), view.byteOffset() + index * Float.BYTES, value);
    }

    private static void checkChannelMajor(
            MemoryView<MemorySegment> view, int channels, int time, String name) {
        Views.requireDense(view, DataType.FP32, name);
        require(
                view.shape().flatRank() == 2
                        && view.shape().flatAt(0) == channels
                        && view.shape().flatAt(1) == time,
                name + " must be [channels, time]");
    }

    private static void checkTimeMajor(
            MemoryView<MemorySegment> view, int steps, int channels, String name) {
        Views.requireDense(view, DataType.FP32, name);
        require(
                view.shape().flatRank() == 2
                        && view.shape().flatAt(0) == steps
                        && view.shape().flatAt(1) == channels,
                name + " must be [steps, channels]");
    }

    private static void require(boolean condition, String message) {
        if (!condition) throw new IllegalArgumentException("Kokoro: " + message);
    }
}
