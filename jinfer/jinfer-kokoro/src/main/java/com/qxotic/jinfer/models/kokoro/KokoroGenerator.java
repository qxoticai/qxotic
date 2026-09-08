package com.qxotic.jinfer.models.kokoro;

import static com.qxotic.jinfer.Segments.readFloat;

import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.kernels.Convert;
import com.qxotic.jinfer.kernels.Convolutions;
import com.qxotic.jinfer.kernels.Ops;
import com.qxotic.jota.DataType;
import com.qxotic.jota.memory.MemoryAllocator;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/** Kokoro's iSTFTNet generator. */
final class KokoroGenerator {

    private static final int STYLE_CHANNELS = 128;
    private static final int SPECTRUM_CHANNELS = 22;
    private static final int BINS = 11;
    private static final int[] DILATIONS = {1, 3, 5};

    record Step(
            KokoroLayers.AdaIN adain1,
            KokoroLayers.Snake snake1,
            KokoroLayers.Conv1d convolution1,
            KokoroLayers.AdaIN adain2,
            KokoroLayers.Snake snake2,
            KokoroLayers.Conv1d convolution2,
            int dilation) {}

    record AdaINResBlock1(List<Step> steps, int channels) {
        AdaINResBlock1 {
            steps = List.copyOf(steps);
            require(steps.size() == 3, "generator residual block must have three steps");
        }

        MemoryView<MemorySegment> forward(
                MemoryView<MemorySegment> input,
                int time,
                MemoryView<MemorySegment> style,
                MemoryAllocator<MemorySegment> scratch) {
            MemoryView<MemorySegment> current = Views.allocateF32(scratch, channels, time);
            Convert.copyF32(input, 0, current, 0, Math.multiplyExact(channels, time));
            for (Step step : steps) {
                try (var ignored = KokoroWorkspace.scope(scratch)) {
                    MemoryView<MemorySegment> branch =
                            step.adain1().forward(current, time, style, scratch);
                    branch = step.snake1().forward(branch, time, scratch);
                    branch = dilated(step.convolution1(), branch, time, step.dilation(), scratch);
                    branch = step.adain2().forward(branch, time, style, scratch);
                    branch = step.snake2().forward(branch, time, scratch);
                    branch = dilated(step.convolution2(), branch, time, 1, scratch);
                    Ops.addInPlace(current, 0, branch, 0, Math.multiplyExact(channels, time));
                }
            }
            return current;
        }
    }

    record Stage(
            KokoroLayers.Conv1d noiseConvolution,
            int noiseStride,
            int noisePadding,
            AdaINResBlock1 noiseResidual,
            KokoroLayers.ConvTranspose1d upsample,
            int upsampleRate,
            int upsamplePadding,
            List<AdaINResBlock1> residuals,
            int channels,
            boolean leftReflectionPad) {
        Stage {
            residuals = List.copyOf(residuals);
        }
    }

    record Weights(List<Stage> stages, KokoroLayers.Conv1d postConvolution) {
        Weights {
            stages = List.copyOf(stages);
        }
    }

    private KokoroGenerator() {}

    static Weights load(
            Map<String, MemoryView<MemorySegment>> tensors,
            Kokoro.Configuration config,
            MemoryAllocator<MemorySegment> persistent) {
        int[] rates = config.upsampleRates();
        int[] kernels = config.upsampleKernelSizes();
        int[] residualKernels = config.resblockKernelSizes();

        List<Stage> stages = new ArrayList<>(2);
        int inputChannels = config.istftInitialChannels();
        for (int stage = 0; stage < 2; stage++) {
            int channels = inputChannels / 2;
            List<AdaINResBlock1> residuals = new ArrayList<>(3);
            for (int block = 0; block < 3; block++)
                residuals.add(
                        block(
                                tensors,
                                persistent,
                                "dec.gen.resblocks." + (stage * 3 + block),
                                residualKernels[block],
                                channels));
            int noiseKernel = stage == 0 ? 12 : 1;
            stages.add(
                    new Stage(
                            KokoroLayers.conv1d(
                                    tensors,
                                    persistent,
                                    "dec.gen.noise_convs." + stage,
                                    noiseKernel,
                                    SPECTRUM_CHANNELS,
                                    channels),
                            stage == 0 ? 6 : 1,
                            stage == 0 ? 3 : 0,
                            block(
                                    tensors,
                                    persistent,
                                    "dec.gen.noise_res." + stage,
                                    stage == 0 ? 7 : 11,
                                    channels),
                            KokoroLayers.convTranspose1d(
                                    tensors,
                                    persistent,
                                    "dec.gen.ups." + stage,
                                    kernels[stage],
                                    inputChannels,
                                    channels),
                            rates[stage],
                            (kernels[stage] - rates[stage]) / 2,
                            residuals,
                            channels,
                            stage == 1));
            inputChannels = channels;
        }
        return new Weights(
                stages,
                KokoroLayers.conv1d(
                        tensors, persistent, "dec.gen.conv_post", 7, inputChannels, 2 * BINS));
    }

    static KokoroDsp.Spectrum forward(
            Weights weights,
            MemoryView<MemorySegment> decoder,
            KokoroDsp.Spectrum harmonic,
            MemoryView<MemorySegment> decoderStyle,
            MemoryAllocator<MemorySegment> scratch) {
        requireMatrix(decoderStyle, 1, STYLE_CHANNELS, "decoder style");
        require(
                decoder != null
                        && decoder.shape().flatRank() == 2
                        && decoder.shape().flatAt(0) == 512,
                "decoder must be [512, 2*T]");
        int decoderTime = Math.toIntExact(decoder.shape().flatAt(1));
        require(decoderTime > 0 && (decoderTime & 1) == 0, "decoder must have positive length 2*T");
        requireMatrix(decoder, 512, decoderTime, "decoder");
        int frames = Math.addExact(Math.multiplyExact(60, decoderTime), 1);
        MemoryView<MemorySegment> source = harmonic(harmonic, frames, scratch);

        MemoryView<MemorySegment> current = decoder;
        int time = decoderTime;
        for (Stage stage : weights.stages()) {
            Ops.leakyReluInPlace(current, 0, Math.toIntExact(current.logicalSize()), 0.1f);
            MemoryView<MemorySegment> injected =
                    stage.noiseConvolution()
                            .forward(
                                    source,
                                    frames,
                                    stage.noiseStride(),
                                    stage.noisePadding(),
                                    scratch);
            int sourceTime = Math.toIntExact(injected.shape().flatAt(1));
            injected = stage.noiseResidual().forward(injected, sourceTime, decoderStyle, scratch);

            current =
                    stage.upsample()
                            .forward(
                                    current,
                                    time,
                                    stage.upsampleRate(),
                                    stage.upsamplePadding(),
                                    0,
                                    scratch);
            time = Math.toIntExact(current.shape().flatAt(1));
            if (stage.leftReflectionPad()) {
                current = reflectionPadLeft(current, stage.channels(), time, scratch);
                time++;
            }
            Ops.addInPlace(current, 0, injected, 0, Math.multiplyExact(stage.channels(), time));

            List<AdaINResBlock1> residuals = stage.residuals();
            MemoryView<MemorySegment> sum =
                    residuals.getFirst().forward(current, time, decoderStyle, scratch);
            int size = Math.multiplyExact(stage.channels(), time);
            for (int i = 1; i < residuals.size(); i++) {
                try (var ignored = KokoroWorkspace.scope(scratch)) {
                    MemoryView<MemorySegment> branch =
                            residuals.get(i).forward(current, time, decoderStyle, scratch);
                    Ops.addInPlace(sum, 0, branch, 0, size);
                }
            }
            Ops.divideInPlace(sum, 0, size, residuals.size());
            current = sum;
        }

        require(time == frames, "generator and harmonic frame lengths differ");
        Ops.leakyReluInPlace(current, 0, Math.toIntExact(current.logicalSize()), 0.01f);
        return outputTransform(
                weights.postConvolution().forward(current, time, scratch), time, scratch);
    }

    private static AdaINResBlock1 block(
            Map<String, MemoryView<MemorySegment>> tensors,
            MemoryAllocator<MemorySegment> persistent,
            String prefix,
            int kernel,
            int channels) {
        List<Step> steps = new ArrayList<>(3);
        for (int step = 0; step < 3; step++) {
            String suffix = "." + step;
            steps.add(
                    new Step(
                            adain(tensors, persistent, prefix + ".adain1" + suffix, channels),
                            new KokoroLayers.Snake(
                                    KokoroLayers.vector(
                                            tensors,
                                            persistent,
                                            prefix + ".alpha1" + suffix,
                                            channels),
                                    channels),
                            KokoroLayers.conv1d(
                                    tensors,
                                    persistent,
                                    prefix + ".convs1" + suffix,
                                    kernel,
                                    channels,
                                    channels),
                            adain(tensors, persistent, prefix + ".adain2" + suffix, channels),
                            new KokoroLayers.Snake(
                                    KokoroLayers.vector(
                                            tensors,
                                            persistent,
                                            prefix + ".alpha2" + suffix,
                                            channels),
                                    channels),
                            KokoroLayers.conv1d(
                                    tensors,
                                    persistent,
                                    prefix + ".convs2" + suffix,
                                    kernel,
                                    channels,
                                    channels),
                            DILATIONS[step]));
        }
        return new AdaINResBlock1(steps, channels);
    }

    private static KokoroLayers.AdaIN adain(
            Map<String, MemoryView<MemorySegment>> tensors,
            MemoryAllocator<MemorySegment> persistent,
            String name,
            int channels) {
        return new KokoroLayers.AdaIN(
                KokoroLayers.linear(tensors, persistent, name, STYLE_CHANNELS, 2 * channels),
                channels);
    }

    private static MemoryView<MemorySegment> dilated(
            KokoroLayers.Conv1d convolution,
            MemoryView<MemorySegment> input,
            int time,
            int dilation,
            MemoryAllocator<MemorySegment> allocator) {
        MemoryView<MemorySegment> output =
                Views.allocateF32(allocator, convolution.outChannels(), time);
        Convolutions.conv1dRows(
                input,
                convolution.inChannels(),
                output,
                convolution.outChannels(),
                time,
                convolution.kernel(),
                dilation,
                convolution.taps(),
                convolution.bias());
        return output;
    }

    static MemoryView<MemorySegment> reflectionPadLeft(
            MemoryView<MemorySegment> input,
            int channels,
            int time,
            MemoryAllocator<MemorySegment> allocator) {
        require(time > 1, "reflection padding requires at least two samples");
        MemoryView<MemorySegment> result = Views.allocateF32(allocator, channels, time + 1);
        for (int channel = 0; channel < channels; channel++) {
            int from = channel * time, to = channel * (time + 1);
            Convert.copyF32(input, from + 1L, result, to, 1);
            Convert.copyF32(input, from, result, to + 1L, time);
        }
        return result;
    }

    static KokoroDsp.Spectrum outputTransform(
            MemoryView<MemorySegment> input, int frames, MemoryAllocator<MemorySegment> scratch) {
        requireMatrix(input, 2 * BINS, frames, "generator output");
        float[][] magnitude = KokoroWorkspace.takeMatrix(scratch, BINS, frames);
        float[][] phase = KokoroWorkspace.takeMatrix(scratch, BINS, frames);
        for (int bin = 0; bin < BINS; bin++)
            for (int frame = 0; frame < frames; frame++) {
                magnitude[bin][frame] = (float) Math.exp(get(input, (long) bin * frames + frame));
                phase[bin][frame] =
                        (float) Math.sin(get(input, (long) (BINS + bin) * frames + frame));
            }
        return new KokoroDsp.Spectrum(magnitude, phase);
    }

    private static MemoryView<MemorySegment> harmonic(
            KokoroDsp.Spectrum harmonic, int frames, MemoryAllocator<MemorySegment> allocator) {
        require(harmonic != null, "harmonic spectrum must not be null");
        require(
                harmonic.magnitude() != null
                        && harmonic.phase() != null
                        && harmonic.magnitude().length == BINS
                        && harmonic.phase().length == BINS,
                "harmonic spectrum must have 11 magnitude and phase bins");
        MemoryView<MemorySegment> result = Views.allocateF32(allocator, SPECTRUM_CHANNELS, frames);
        for (int bin = 0; bin < BINS; bin++) {
            require(
                    harmonic.magnitude()[bin].length == frames
                            && harmonic.phase()[bin].length == frames,
                    "harmonic spectrum has the wrong frame count");
            Views.copyFromArray(
                    result, (long) bin * frames, harmonic.magnitude()[bin], 0, frames, "harmonic");
            Views.copyFromArray(
                    result,
                    (long) (BINS + bin) * frames,
                    harmonic.phase()[bin],
                    0,
                    frames,
                    "harmonic");
        }
        return result;
    }

    private static float get(MemoryView<MemorySegment> view, long index) {
        return readFloat(view.memory().base(), view.byteOffset() + index * Float.BYTES);
    }

    private static void requireMatrix(
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
