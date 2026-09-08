package com.qxotic.jinfer.models.kokoro;

import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.kernels.Activations;
import com.qxotic.jinfer.kernels.Convert;
import com.qxotic.jinfer.kernels.FlashAttention;
import com.qxotic.jinfer.kernels.MatMul;
import com.qxotic.jinfer.kernels.Norms;
import com.qxotic.jinfer.kernels.Ops;
import com.qxotic.jota.memory.MemoryAllocator;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;
import java.util.Map;

/** Kokoro's parameter-shared PL-BERT/ALBERT encoder and 512-channel projection. */
final class PlBert {

    private static final float LAYER_NORM_EPS = 1e-12f;

    record Layer(
            KokoroLayers.Linear query,
            KokoroLayers.Linear key,
            KokoroLayers.Linear value,
            KokoroLayers.Linear attentionOutput,
            MemoryView<MemorySegment> attentionGamma,
            MemoryView<MemorySegment> attentionBeta,
            KokoroLayers.Linear ffnUp,
            KokoroLayers.Linear ffnDown,
            MemoryView<MemorySegment> ffnGamma,
            MemoryView<MemorySegment> ffnBeta) {}

    record Weights(
            MemoryView<MemorySegment> tokenEmbedding,
            MemoryView<MemorySegment> positionEmbedding,
            MemoryView<MemorySegment> tokenType,
            MemoryView<MemorySegment> embeddingGamma,
            MemoryView<MemorySegment> embeddingBeta,
            KokoroLayers.Linear embeddingProjection,
            Layer layer,
            KokoroLayers.Linear outputProjection) {}

    private PlBert() {}

    static Weights load(
            Map<String, MemoryView<MemorySegment>> tensors,
            Kokoro.Configuration config,
            MemoryAllocator<MemorySegment> allocator) {
        int embedding = config.plbertEmbeddingSize();
        int hidden = config.plbertHiddenSize();
        int intermediate = config.plbertIntermediateSize();

        MemoryView<MemorySegment> tokenEmbedding =
                KokoroLayers.matrix(
                        tensors, "bert.embd.tok.weight", config.tokenCount(), embedding);
        MemoryView<MemorySegment> positionEmbedding =
                KokoroLayers.matrix(
                        tensors, "bert.embd.pos.weight", config.plbertMaxPositions(), embedding);
        MemoryView<MemorySegment> tokenTypes =
                KokoroLayers.matrix(tensors, "bert.embd.tt.weight", 2, embedding);
        MemoryView<MemorySegment> tokenType = Views.allocateF32(allocator, embedding);
        Convert.copyToF32(tokenTypes, 0, tokenType, 0, embedding);

        Layer layer =
                new Layer(
                        KokoroLayers.linear(tensors, allocator, "bert.attn_q", hidden, hidden),
                        KokoroLayers.linear(tensors, allocator, "bert.attn_k", hidden, hidden),
                        KokoroLayers.linear(tensors, allocator, "bert.attn_v", hidden, hidden),
                        KokoroLayers.linear(tensors, allocator, "bert.attn_o", hidden, hidden),
                        KokoroLayers.vector(tensors, allocator, "bert.attn_ln.weight", hidden),
                        KokoroLayers.vector(tensors, allocator, "bert.attn_ln.bias", hidden),
                        KokoroLayers.linear(
                                tensors, allocator, "bert.ffn_up", hidden, intermediate),
                        KokoroLayers.linear(
                                tensors, allocator, "bert.ffn_down", intermediate, hidden),
                        KokoroLayers.vector(tensors, allocator, "bert.ffn_ln.weight", hidden),
                        KokoroLayers.vector(tensors, allocator, "bert.ffn_ln.bias", hidden));
        return new Weights(
                tokenEmbedding,
                positionEmbedding,
                tokenType,
                KokoroLayers.vector(tensors, allocator, "bert.embd.ln.weight", embedding),
                KokoroLayers.vector(tensors, allocator, "bert.embd.ln.bias", embedding),
                KokoroLayers.linear(tensors, allocator, "bert.embd_proj", embedding, hidden),
                layer,
                KokoroLayers.linear(tensors, allocator, "bert_proj", hidden, config.hiddenDim()));
    }

    /** Returns contiguous FP32 {@code [steps, 512]}; callers supply any boundary pad tokens. */
    static MemoryView<MemorySegment> encode(
            Weights weights,
            int[] tokens,
            Kokoro.Configuration config,
            MemoryAllocator<MemorySegment> scratch) {
        require(tokens.length > 0, "PL-BERT input is empty");
        require(tokens.length <= config.plbertMaxPositions(), "PL-BERT input is too long");
        for (int token : tokens) {
            require(token >= 0 && token < config.tokenCount(), "invalid token " + token);
        }

        int steps = tokens.length;
        int embedding = config.plbertEmbeddingSize();
        int hidden = config.plbertHiddenSize();
        int elements = Math.multiplyExact(steps, embedding);
        MemoryView<MemorySegment> embedded = Views.allocateF32(scratch, steps, embedding);
        MemoryView<MemorySegment> positions = Views.allocateF32(scratch, steps, embedding);
        Convert.gatherToF32(weights.tokenEmbedding(), tokens, 0, steps, embedded, 0, embedding);
        Convert.copyToF32(weights.positionEmbedding(), 0, positions, 0, elements);
        Ops.addInPlace(embedded, 0, positions, 0, elements);
        Ops.addRowBiasInPlace(embedded, 0, weights.tokenType(), 0, steps, embedding);
        Norms.layerNorm(
                embedded,
                embedded,
                weights.embeddingGamma(),
                weights.embeddingBeta(),
                embedding,
                steps,
                LAYER_NORM_EPS);

        MemoryView<MemorySegment> current = Views.allocateF32(scratch, steps, hidden);
        linear(weights.embeddingProjection(), embedded, current, steps);
        transform(
                weights.layer(),
                current,
                config.plbertHeads(),
                config.plbertIntermediateSize(),
                config.plbertLayers(),
                scratch);

        MemoryView<MemorySegment> output =
                Views.allocateF32(scratch, steps, weights.outputProjection().outputSize());
        linear(weights.outputProjection(), current, output, steps);
        return output;
    }

    /**
     * Parameterized shared-layer seam used by the fixed-layout encoder and small contract tests.
     */
    static void transform(
            Layer layer,
            MemoryView<MemorySegment> current,
            int heads,
            int intermediate,
            int repeats,
            MemoryAllocator<MemorySegment> scratch) {
        require(current.shape().flatRank() == 2, "ALBERT input must be [steps, hidden]");
        int steps = Math.toIntExact(current.shape().flatAt(0));
        int hidden = Math.toIntExact(current.shape().flatAt(1));
        require(heads > 0 && hidden % heads == 0, "ALBERT hidden size must divide into heads");
        require(intermediate > 0 && repeats > 0, "invalid ALBERT dimensions");
        validateLayer(layer, hidden, intermediate);

        MemoryView<MemorySegment> query = Views.allocateF32(scratch, steps, hidden);
        MemoryView<MemorySegment> key = Views.allocateF32(scratch, steps, hidden);
        MemoryView<MemorySegment> value = Views.allocateF32(scratch, steps, hidden);
        MemoryView<MemorySegment> temporary = Views.allocateF32(scratch, steps, hidden);
        MemoryView<MemorySegment> ffn = Views.allocateF32(scratch, steps, intermediate);
        int elements = Math.multiplyExact(steps, hidden);
        int headSize = hidden / heads;
        float scale = 1f / (float) Math.sqrt(headSize);

        for (int repeat = 0; repeat < repeats; repeat++) {
            linear(layer.query(), current, query, steps);
            linear(layer.key(), current, key, steps);
            linear(layer.value(), current, value, steps);
            FlashAttention.bidirectionalPrefill(
                    query, temporary, key, value, heads, steps, headSize, hidden, hidden, 1, scale);
            linear(layer.attentionOutput(), temporary, query, steps);
            Ops.addInPlace(current, 0, query, 0, elements);
            Norms.layerNorm(
                    current,
                    current,
                    layer.attentionGamma(),
                    layer.attentionBeta(),
                    hidden,
                    steps,
                    LAYER_NORM_EPS);

            linear(layer.ffnUp(), current, ffn, steps);
            Activations.geluInPlace(ffn, 0, Math.multiplyExact(steps, intermediate));
            linear(layer.ffnDown(), ffn, temporary, steps);
            Ops.addInPlace(current, 0, temporary, 0, elements);
            Norms.layerNorm(
                    current,
                    current,
                    layer.ffnGamma(),
                    layer.ffnBeta(),
                    hidden,
                    steps,
                    LAYER_NORM_EPS);
        }
    }

    private static void linear(
            KokoroLayers.Linear linear,
            MemoryView<MemorySegment> input,
            MemoryView<MemorySegment> output,
            int rows) {
        MatMul.gemm(
                linear.weight(),
                input,
                linear.inputSize(),
                output,
                linear.outputSize(),
                linear.outputSize(),
                rows,
                linear.inputSize());
        Ops.addRowBiasInPlace(output, 0, linear.bias(), 0, rows, linear.outputSize());
    }

    private static void validateLayer(Layer layer, int hidden, int intermediate) {
        validate(layer.query(), hidden, hidden, "query");
        validate(layer.key(), hidden, hidden, "key");
        validate(layer.value(), hidden, hidden, "value");
        validate(layer.attentionOutput(), hidden, hidden, "attention output");
        require(layer.attentionGamma().logicalSize() == hidden, "attention gamma has wrong size");
        require(layer.attentionBeta().logicalSize() == hidden, "attention beta has wrong size");
        validate(layer.ffnUp(), hidden, intermediate, "FFN up");
        validate(layer.ffnDown(), intermediate, hidden, "FFN down");
        require(layer.ffnGamma().logicalSize() == hidden, "FFN gamma has wrong size");
        require(layer.ffnBeta().logicalSize() == hidden, "FFN beta has wrong size");
    }

    private static void validate(KokoroLayers.Linear linear, int input, int output, String name) {
        require(
                linear.inputSize() == input && linear.outputSize() == output,
                name + " has wrong dimensions");
    }

    private static void require(boolean condition, String message) {
        if (!condition) throw new IllegalArgumentException("Kokoro: " + message);
    }
}
