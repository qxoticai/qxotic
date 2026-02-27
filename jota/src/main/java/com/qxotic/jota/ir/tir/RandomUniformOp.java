package com.qxotic.jota.ir.tir;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;

/**
 * Stateless random-uniform tensor node.
 *
 * <p>Values are deterministically derived from {@code key0,key1} and output element index.
 */
public record RandomUniformOp(Shape shape, DataType dataType, long key0, long key1)
        implements TIRNode {

    public RandomUniformOp {
        if (shape == null) {
            throw new IllegalArgumentException("shape cannot be null");
        }
        if (dataType != DataType.FP32 && dataType != DataType.FP64) {
            throw new IllegalArgumentException(
                    "RandomUniformOp supports FP32/FP64 only, got: " + dataType);
        }
    }
}
