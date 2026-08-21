package com.qxotic.jinfer.models.gemma4;

import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.kernels.Convert;
import com.qxotic.jinfer.kernels.MatMul;
import com.qxotic.jinfer.kernels.Ops;
import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;
import java.util.Map;

/** A projection with optional Gemma 4 QAT input and output clamps. */
record Clamped(
        MemoryView<MemorySegment> weight,
        float inputMin,
        float inputMax,
        float outputMin,
        float outputMax) {

    static Clamped load(
            Map<String, MemoryView<MemorySegment>> tensors, String base, int outDim, int inDim) {
        return new Clamped(
                Gemma4VisionUnified.requireWeight(
                        tensors, base + ".weight", Shape.flat(outDim, inDim)),
                scalar(tensors, base + ".input_min", -Float.MAX_VALUE),
                scalar(tensors, base + ".input_max", Float.MAX_VALUE),
                scalar(tensors, base + ".output_min", -Float.MAX_VALUE),
                scalar(tensors, base + ".output_max", Float.MAX_VALUE));
    }

    private static float scalar(
            Map<String, MemoryView<MemorySegment>> tensors, String name, float defaultValue) {
        MemoryView<MemorySegment> value = tensors.get(name);
        if (value == null) return defaultValue;
        Gemma4VisionUnified.requireF32(value, name, Shape.flat(1));
        return Views.getFloat(value, 0, name);
    }

    void gemm(
            MemoryView<MemorySegment> input,
            int inDim,
            MemoryView<MemorySegment> output,
            int outDim,
            int rows,
            MemoryView<MemorySegment> clampScratch) {
        requireActivation(input, "input", rows, inDim);
        requireActivation(output, "output", rows, outDim);
        MemoryView<MemorySegment> source = input;
        int inputElements = Math.multiplyExact(rows, inDim);
        if (inputMin > -Float.MAX_VALUE || inputMax < Float.MAX_VALUE) {
            requireActivation(clampScratch, "clampScratch", rows, inDim);
            Convert.copyF32(input, 0, clampScratch, 0, inputElements);
            Ops.clampInPlace(clampScratch, 0, inputElements, inputMin, inputMax);
            // ponytail: the clamp scratch is a shared max-width buffer ([count, 3072] here), and
            // MatMul strides A by stride()[0] - not by the contraction width. Passing the raw
            // scratch would read row r at r*3072 while the copy packed rows at r*inDim: row 0
            // correct, every row >= 1 garbage (the docstring's "max-width-scratch MoE trap").
            // Re-view the live prefix at the LOGICAL [rows, inDim] so the stride matches.
            MemorySegment prefix =
                    clampScratch
                            .memory()
                            .base()
                            .asSlice(clampScratch.byteOffset(), (long) inputElements * Float.BYTES);
            source = Views.wrap(prefix, DataType.FP32, Shape.flat(rows, inDim));
        }
        MatMul.gemm(weight, source, output, rows);
        if (outputMin > -Float.MAX_VALUE || outputMax < Float.MAX_VALUE)
            Ops.clampInPlace(output, 0, Math.multiplyExact(rows, outDim), outputMin, outputMax);
    }

    private static void requireActivation(
            MemoryView<MemorySegment> view, String name, int rows, int columns) {
        Views.requireDense(view, DataType.FP32, name);
        long required = Math.multiplyExact(rows, columns);
        if (view.shape().size() < required)
            throw new IllegalArgumentException(
                    name
                            + ": requires at least "
                            + required
                            + " elements but shape was "
                            + view.shape());
    }
}
