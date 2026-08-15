package com.qxotic.jota.ir.tir;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import java.util.Objects;

/** Cast operation node in IR-T. */
public record CastOp(TIRNode input, DataType targetDataType, Shape shape) implements TIRNode {

    public CastOp(TIRNode input, DataType targetDataType) {
        this(input, targetDataType, input.shape());
    }

    public CastOp {
        Objects.requireNonNull(input);
        Objects.requireNonNull(targetDataType);
        Objects.requireNonNull(shape);
    }

    @Override
    public DataType dataType() {
        return targetDataType;
    }

    @Override
    public Shape shape() {
        return shape;
    }
}
