package com.qxotic.jota.ir.tir;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;

/**
 * Iota constant (0..n-1 sequence) for arange support. Only supports base case: [0, 1, 2, ..., n-1]
 * Other variants (start, step) can be expressed as: add(multiply(IotaConstant, step), start)
 */
public record IotaConstant(long count, DataType dataType, Shape shape) implements TIRNode {

    public IotaConstant {
        if (count < 0) {
            throw new IllegalArgumentException("count must be non-negative, got: " + count);
        }
        if (shape == null) {
            throw new IllegalArgumentException("shape cannot be null");
        }
    }

    /** Creates an iota constant with the specified count. */
    public static IotaConstant of(long count, DataType dataType) {
        return new IotaConstant(count, dataType, Shape.flat(count));
    }

    @Override
    public Shape shape() {
        return shape;
    }
}
