package com.qxotic.jinfer.models.kokoro;

import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.kernels.Convert;
import com.qxotic.jinfer.kernels.Ops;
import com.qxotic.jota.DataType;
import com.qxotic.jota.memory.MemoryAllocator;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/** Kokoro's pre-generator decoder body. */
final class KokoroDecoder {

    private static final int TEXT_CHANNELS = 512;
    private static final int STYLE_CHANNELS = 128;
    private static final int ENCODED_CHANNELS = 1024;
    private static final int ASR_RESIDUAL_CHANNELS = 64;
    private static final int CONDITIONED_CHANNELS = 1090;

    record Weights(
            KokoroLayers.Conv1d f0Convolution,
            KokoroLayers.Conv1d noiseConvolution,
            KokoroLayers.Conv1d asrResidual,
            KokoroLayers.AdainResBlock encoder,
            List<KokoroLayers.AdainResBlock> decoder) {
        Weights {
            decoder = List.copyOf(decoder);
        }
    }

    private KokoroDecoder() {}

    static Weights load(
            Map<String, MemoryView<MemorySegment>> tensors,
            MemoryAllocator<MemorySegment> persistent) {
        List<KokoroLayers.AdainResBlock> decoder = new ArrayList<>(4);
        for (int layer = 0; layer < 4; layer++) {
            decoder.add(
                    KokoroLayers.adainResBlock(
                            tensors,
                            persistent,
                            "dec.decode." + layer,
                            CONDITIONED_CHANNELS,
                            layer == 3 ? TEXT_CHANNELS : ENCODED_CHANNELS,
                            STYLE_CHANNELS,
                            layer == 3));
        }
        return new Weights(
                KokoroLayers.conv1d(tensors, persistent, "dec.F0_conv", 3, 1, 1),
                KokoroLayers.conv1d(tensors, persistent, "dec.N_conv", 3, 1, 1),
                KokoroLayers.conv1d(
                        tensors,
                        persistent,
                        "dec.asr_res",
                        1,
                        TEXT_CHANNELS,
                        ASR_RESIDUAL_CHANNELS),
                KokoroLayers.adainResBlock(
                        tensors,
                        persistent,
                        "dec.encode",
                        TEXT_CHANNELS + 2,
                        ENCODED_CHANNELS,
                        STYLE_CHANNELS,
                        false),
                decoder);
    }

    static MemoryView<MemorySegment> forward(
            Weights weights,
            MemoryView<MemorySegment> textEncoding,
            int[] alignment,
            float[] f0,
            float[] noise,
            MemoryView<MemorySegment> decoderStyle,
            MemoryAllocator<MemorySegment> scratch) {
        require(alignment.length > 0, "decoder alignment is empty");
        int time = alignment.length;
        require(f0.length == 2 * time && noise.length == 2 * time, "invalid decoder curves");
        checkMatrix(decoderStyle, 1, STYLE_CHANNELS, "decoder style");

        MemoryView<MemorySegment> asr = gatherAsr(textEncoding, alignment, scratch);
        MemoryView<MemorySegment> f0Down =
                downsampleCurve(weights.f0Convolution(), f0, time, scratch);
        MemoryView<MemorySegment> noiseDown =
                downsampleCurve(weights.noiseConvolution(), noise, time, scratch);
        MemoryView<MemorySegment> current =
                concatenateConditioning(asr, TEXT_CHANNELS, f0Down, noiseDown, time, scratch);
        current = weights.encoder().forward(current, time, decoderStyle, scratch);

        MemoryView<MemorySegment> asrResidual = weights.asrResidual().forward(asr, time, scratch);
        for (KokoroLayers.AdainResBlock block : weights.decoder()) {
            current =
                    concatenateConditioning(
                            current,
                            ENCODED_CHANNELS,
                            asrResidual,
                            ASR_RESIDUAL_CHANNELS,
                            f0Down,
                            noiseDown,
                            time,
                            scratch);
            current = block.forward(current, time, decoderStyle, scratch);
        }
        return current;
    }

    static MemoryView<MemorySegment> gatherAsr(
            MemoryView<MemorySegment> textEncoding,
            int[] alignment,
            MemoryAllocator<MemorySegment> allocator) {
        MemoryView<MemorySegment> gathered =
                ProsodyPredictor.gatherRows(textEncoding, alignment, TEXT_CHANNELS, allocator);
        MemoryView<MemorySegment> channelMajor =
                Views.allocateF32(allocator, TEXT_CHANNELS, alignment.length);
        Ops.transposeCopy(gathered, alignment.length, TEXT_CHANNELS, channelMajor);
        return channelMajor;
    }

    static MemoryView<MemorySegment> downsampleCurve(
            KokoroLayers.Conv1d convolution,
            float[] curve,
            int outputTime,
            MemoryAllocator<MemorySegment> allocator) {
        require(curve.length == 2 * outputTime, "invalid decoder curve length");
        MemoryView<MemorySegment> input = Views.allocateF32(allocator, 1, curve.length);
        Views.copyFromArray(input, 0, curve, 0, curve.length, "decoder curve");
        return convolution.forward(input, curve.length, 2, 1, allocator);
    }

    static MemoryView<MemorySegment> concatenateConditioning(
            MemoryView<MemorySegment> input,
            int inputChannels,
            MemoryView<MemorySegment> f0,
            MemoryView<MemorySegment> noise,
            int time,
            MemoryAllocator<MemorySegment> allocator) {
        int inputSize = Math.multiplyExact(inputChannels, time);
        MemoryView<MemorySegment> result = Views.allocateF32(allocator, inputChannels + 2, time);
        Convert.copyF32(input, 0, result, 0, inputSize);
        Convert.copyF32(f0, 0, result, inputSize, time);
        Convert.copyF32(noise, 0, result, inputSize + time, time);
        return result;
    }

    static MemoryView<MemorySegment> concatenateConditioning(
            MemoryView<MemorySegment> input,
            int inputChannels,
            MemoryView<MemorySegment> asrResidual,
            int asrChannels,
            MemoryView<MemorySegment> f0,
            MemoryView<MemorySegment> noise,
            int time,
            MemoryAllocator<MemorySegment> allocator) {
        int inputSize = Math.multiplyExact(inputChannels, time);
        int asrSize = Math.multiplyExact(asrChannels, time);
        MemoryView<MemorySegment> result =
                Views.allocateF32(allocator, inputChannels + asrChannels + 2, time);
        Convert.copyF32(input, 0, result, 0, inputSize);
        Convert.copyF32(asrResidual, 0, result, inputSize, asrSize);
        Convert.copyF32(f0, 0, result, inputSize + asrSize, time);
        Convert.copyF32(noise, 0, result, inputSize + asrSize + time, time);
        return result;
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
