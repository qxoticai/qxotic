package com.qxotic.jinfer.models.kokoro;

import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.kernels.Convert;
import com.qxotic.jinfer.kernels.Convolutions;
import com.qxotic.jinfer.kernels.ModelLoader;
import com.qxotic.jinfer.kernels.Norms;
import com.qxotic.jinfer.kernels.Ops;
import com.qxotic.jota.memory.MemoryAllocator;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/** Kokoro's embedding, convolution stack, and bidirectional LSTM. */
final class TextEncoder {

    private static final float LAYER_NORM_EPS = 1e-5f;
    private static final float LEAKY_RELU_SLOPE = 0.2f;

    record Conv(
            float[] taps,
            MemoryView<MemorySegment> bias,
            MemoryView<MemorySegment> gamma,
            MemoryView<MemorySegment> beta) {}

    record Weights(
            MemoryView<MemorySegment> embedding,
            List<Conv> convolutions,
            KokoroOps.LstmWeights forward,
            KokoroOps.LstmWeights reverse) {
        Weights {
            convolutions = List.copyOf(convolutions);
        }
    }

    private TextEncoder() {}

    static Weights load(
            Map<String, MemoryView<MemorySegment>> tensors,
            Kokoro.Configuration config,
            MemoryAllocator<MemorySegment> allocator) {
        int hidden = config.hiddenDim();
        MemoryView<MemorySegment> embedding = ModelLoader.require(tensors, "text_enc.embd.weight");
        require(
                embedding.logicalSize() % config.tokenCount() == 0,
                "text_enc.embd.weight cannot be split into token rows");
        int embeddingStride = Math.toIntExact(embedding.logicalSize() / config.tokenCount());
        require(embeddingStride == hidden, "text_enc.embd.weight has the wrong row width");

        List<Conv> convolutions = new ArrayList<>(3);
        for (int layer = 0; layer < 3; layer++) {
            String prefix = "text_enc.cnn." + layer;
            convolutions.add(
                    new Conv(
                            KokoroLayers.convolutionTaps(
                                    tensors,
                                    allocator,
                                    prefix + ".conv.weight",
                                    config.textEncoderKernelSize(),
                                    hidden,
                                    hidden),
                            KokoroLayers.vector(tensors, allocator, prefix + ".conv.bias", hidden),
                            KokoroLayers.vector(tensors, allocator, prefix + ".ln.gamma", hidden),
                            KokoroLayers.vector(tensors, allocator, prefix + ".ln.beta", hidden)));
        }
        return new Weights(
                embedding,
                convolutions,
                KokoroLayers.lstm(tensors, allocator, "text_enc.lstm", "", hidden, hidden / 2),
                KokoroLayers.lstm(
                        tensors, allocator, "text_enc.lstm", "_reverse", hidden, hidden / 2));
    }

    /**
     * Returns contiguous {@code [steps, hidden]} output, matching GGML's physical (hidden, steps).
     */
    static MemoryView<MemorySegment> encode(
            Weights weights,
            int[] tokens,
            int hidden,
            int kernel,
            MemoryAllocator<MemorySegment> scratch) {
        require(tokens.length > 0, "text encoder input is empty");
        for (int token : tokens) {
            require(
                    token >= 0 && token < weights.embedding().shape().flatAt(0),
                    "invalid token " + token);
        }

        int steps = tokens.length;
        MemoryView<MemorySegment> timeMajor = Views.allocateF32(scratch, steps, hidden);
        MemoryView<MemorySegment> next = Views.allocateF32(scratch, steps, hidden);
        Convert.gatherToF32(weights.embedding(), tokens, 0, steps, timeMajor, 0, hidden);

        for (Conv convolution : weights.convolutions()) {
            try (var ignored = KokoroWorkspace.scope(scratch)) {
                MemoryView<MemorySegment> channelMajor = Views.allocateF32(scratch, hidden, steps);
                Ops.transposeCopy(timeMajor, steps, hidden, channelMajor);
                MemoryView<MemorySegment> convolved = Views.allocateF32(scratch, hidden, steps);
                Convolutions.conv1dRows(
                        channelMajor,
                        hidden,
                        convolved,
                        hidden,
                        steps,
                        kernel,
                        1,
                        convolution.taps(),
                        convolution.bias());
                Ops.transposeCopy(convolved, hidden, steps, next);
            }
            Norms.layerNorm(
                    next,
                    next,
                    convolution.gamma(),
                    convolution.beta(),
                    hidden,
                    steps,
                    LAYER_NORM_EPS);
            Ops.leakyReluInPlace(next, 0, Math.multiplyExact(steps, hidden), LEAKY_RELU_SLOPE);
            MemoryView<MemorySegment> swap = timeMajor;
            timeMajor = next;
            next = swap;
        }

        MemoryView<MemorySegment> output = Views.allocateF32(scratch, steps, hidden);
        KokoroOps.bidirectionalLstm(
                timeMajor, weights.forward(), weights.reverse(), output, scratch);
        return output;
    }

    private static void require(boolean condition, String message) {
        if (!condition) throw new IllegalArgumentException("Kokoro: " + message);
    }
}
