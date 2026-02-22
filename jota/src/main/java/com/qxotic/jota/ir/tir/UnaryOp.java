package com.qxotic.jota.ir.tir;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import java.util.Objects;

/** Unary operation node in IR-T. */
public record UnaryOp(UnaryOperator op, TIRNode input, Shape shape) implements TIRNode {

    public UnaryOp(UnaryOperator op, TIRNode input) {
        this(op, input, input.shape());
    }

    public UnaryOp {
        Objects.requireNonNull(op);
        Objects.requireNonNull(input);
        Objects.requireNonNull(shape);
    }

    @Override
    public DataType dataType() {
        return input.dataType();
    }

    @Override
    public Shape shape() {
        return shape;
    }
}
