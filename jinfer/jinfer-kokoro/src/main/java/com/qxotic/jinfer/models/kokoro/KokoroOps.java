package com.qxotic.jinfer.models.kokoro;

import static com.qxotic.jinfer.Segments.readFloat;
import static com.qxotic.jinfer.Segments.writeFloat;

import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.kernels.MatMul;
import com.qxotic.jota.DataType;
import com.qxotic.jota.memory.MemoryAllocator;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;

/** Model-private operations missing from the shared Jinfer kernels. */
final class KokoroOps {

    private KokoroOps() {}

    record LstmWeights(
            MemoryView<MemorySegment> input,
            MemoryView<MemorySegment> hidden,
            MemoryView<MemorySegment> inputBias,
            MemoryView<MemorySegment> hiddenBias) {}

    static void bidirectionalLstm(
            MemoryView<MemorySegment> input,
            LstmWeights forwardWeights,
            LstmWeights reverseWeights,
            MemoryView<MemorySegment> output,
            MemoryAllocator<MemorySegment> scratch) {
        require(output.shape().flatRank() == 2, "output must be [steps, 2*hidden_size]");
        int steps = Math.toIntExact(input.shape().flatAt(0));
        int hiddenSize = weightColumns(forwardWeights.hidden);
        require(
                output.shape().flatAt(0) == steps && output.shape().flatAt(1) == 2L * hiddenSize,
                "bidirectional output must be [steps, 2*hidden_size]");
        MemoryView<MemorySegment> forward = Views.allocateF32(scratch, steps, hiddenSize);
        MemoryView<MemorySegment> reverse = Views.allocateF32(scratch, steps, hiddenSize);
        lstm(
                input,
                forwardWeights.input,
                forwardWeights.hidden,
                forwardWeights.inputBias,
                forwardWeights.hiddenBias,
                false,
                forward,
                scratch);
        lstm(
                input,
                reverseWeights.input,
                reverseWeights.hidden,
                reverseWeights.inputBias,
                reverseWeights.hiddenBias,
                true,
                reverse,
                scratch);
        for (int step = 0; step < steps; step++) {
            for (int channel = 0; channel < hiddenSize; channel++) {
                set(
                        output,
                        (long) step * 2 * hiddenSize + channel,
                        get(forward, (long) step * hiddenSize + channel));
                set(
                        output,
                        (long) step * 2 * hiddenSize + hiddenSize + channel,
                        get(reverse, (long) step * hiddenSize + channel));
            }
        }
    }

    /** PyTorch LSTM gate order: input, forget, cell candidate, output. */
    static void lstm(
            MemoryView<MemorySegment> input,
            MemoryView<MemorySegment> weightInput,
            MemoryView<MemorySegment> weightHidden,
            MemoryView<MemorySegment> biasInput,
            MemoryView<MemorySegment> biasHidden,
            boolean reverse,
            MemoryView<MemorySegment> output,
            MemoryAllocator<MemorySegment> scratch) {
        Views.requireDense(input, DataType.FP32, "lstm input");
        Views.requireDense(output, DataType.FP32, "lstm output");
        Views.requireDense(biasInput, DataType.FP32, "lstm input bias");
        Views.requireDense(biasHidden, DataType.FP32, "lstm hidden bias");
        Views.requireContiguous(weightInput, "lstm input weight");
        Views.requireContiguous(weightHidden, "lstm hidden weight");

        require(input.shape().flatRank() == 2, "input must be [steps, input_size]");
        require(output.shape().flatRank() == 2, "output must be [steps, hidden_size]");
        int steps = Math.toIntExact(input.shape().flatAt(0));
        int inputSize = Math.toIntExact(input.shape().flatAt(1));
        int hiddenSize = Math.toIntExact(output.shape().flatAt(1));
        int gates = Math.multiplyExact(hiddenSize, 4);
        require(output.shape().flatAt(0) == steps, "input and output step counts differ");
        require(
                weightRows(weightInput) == gates && weightColumns(weightInput) == inputSize,
                "input weight must be [4*hidden_size, input_size]");
        require(
                weightRows(weightHidden) == gates && weightColumns(weightHidden) == hiddenSize,
                "hidden weight must be [4*hidden_size, hidden_size]");
        require(
                biasInput.logicalSize() == gates && biasHidden.logicalSize() == gates,
                "biases must contain 4*hidden_size values");

        try (var ignored = KokoroWorkspace.scope(scratch)) {
            MemoryView<MemorySegment> projected = Views.allocateF32(scratch, steps, gates);
            MemoryView<MemorySegment> hidden = Views.allocateF32(scratch, 1, hiddenSize);
            MemoryView<MemorySegment> recurrent = Views.allocateF32(scratch, 1, gates);
            hidden.memory()
                    .base()
                    .asSlice(hidden.byteOffset(), (long) hiddenSize * Float.BYTES)
                    .fill((byte) 0);
            float[] cell = KokoroWorkspace.takeFloats(scratch, hiddenSize);
            java.util.Arrays.fill(cell, 0);

            MatMul.gemm(weightInput, input, projected, steps);
            for (int iteration = 0; iteration < steps; iteration++) {
                int step = reverse ? steps - iteration - 1 : iteration;
                MatMul.gemv(weightHidden, hidden, recurrent);
                for (int channel = 0; channel < hiddenSize; channel++) {
                    float inputGate =
                            sigmoid(
                                    gate(
                                            projected,
                                            recurrent,
                                            biasInput,
                                            biasHidden,
                                            step,
                                            gates,
                                            channel));
                    float forgetGate =
                            sigmoid(
                                    gate(
                                            projected,
                                            recurrent,
                                            biasInput,
                                            biasHidden,
                                            step,
                                            gates,
                                            hiddenSize + channel));
                    float candidate =
                            (float)
                                    Math.tanh(
                                            gate(
                                                    projected,
                                                    recurrent,
                                                    biasInput,
                                                    biasHidden,
                                                    step,
                                                    gates,
                                                    2 * hiddenSize + channel));
                    float outputGate =
                            sigmoid(
                                    gate(
                                            projected,
                                            recurrent,
                                            biasInput,
                                            biasHidden,
                                            step,
                                            gates,
                                            3 * hiddenSize + channel));
                    cell[channel] = forgetGate * cell[channel] + inputGate * candidate;
                    float value = outputGate * (float) Math.tanh(cell[channel]);
                    set(hidden, channel, value);
                    set(output, (long) step * hiddenSize + channel, value);
                }
            }
        }
    }

    static int[] durations(
            MemoryView<MemorySegment> logits,
            int tokenCount,
            int maxDuration,
            double speed,
            MemoryAllocator<MemorySegment> scratch) {
        require(
                logits.logicalSize() == Math.multiplyExact(tokenCount, maxDuration),
                "duration logits have the wrong size");
        require(Double.isFinite(speed) && speed > 0, "speed must be finite and positive");
        int[] durations = KokoroWorkspace.takeInts(scratch, tokenCount);
        for (int token = 0; token < tokenCount; token++) {
            float duration = 0;
            int offset = token * maxDuration;
            for (int i = 0; i < maxDuration; i++) {
                float logit = get(logits, offset + i);
                require(Float.isFinite(logit), "duration logits must be finite");
                duration += sigmoid(logit);
            }
            durations[token] = Math.max(1, Math.toIntExact((long) Math.rint(duration / speed)));
        }
        return durations;
    }

    private static float gate(
            MemoryView<MemorySegment> projected,
            MemoryView<MemorySegment> recurrent,
            MemoryView<MemorySegment> biasInput,
            MemoryView<MemorySegment> biasHidden,
            int step,
            int stride,
            int gate) {
        return get(projected, (long) step * stride + gate)
                + get(recurrent, gate)
                + get(biasInput, gate)
                + get(biasHidden, gate);
    }

    private static int weightRows(MemoryView<MemorySegment> weight) {
        require(weight.shape().flatRank() == 2, "LSTM weights must be two-dimensional");
        return Math.toIntExact(weight.shape().flatAt(0));
    }

    private static int weightColumns(MemoryView<MemorySegment> weight) {
        return Math.toIntExact(weight.shape().flatAt(1) * weight.dataType().elementsPerBlock());
    }

    private static float sigmoid(double value) {
        return (float) (1.0 / (1.0 + Math.exp(-value)));
    }

    private static float get(MemoryView<MemorySegment> view, long index) {
        return readFloat(view.memory().base(), view.byteOffset() + index * Float.BYTES);
    }

    private static void set(MemoryView<MemorySegment> view, long index, float value) {
        writeFloat(view.memory().base(), view.byteOffset() + index * Float.BYTES, value);
    }

    private static void require(boolean condition, String message) {
        if (!condition) throw new IllegalArgumentException("Kokoro: " + message);
    }
}
