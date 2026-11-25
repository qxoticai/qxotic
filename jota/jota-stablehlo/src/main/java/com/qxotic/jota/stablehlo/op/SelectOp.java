package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Value;

/** Typed view of {@code stablehlo.select}. Generated from the op table; see {@link Op}. */
public record SelectOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.select";

    public SelectOp {
        OpSupport.checkName(operation, NAME);
    }

    public Value pred() {
        return operation.operand(0);
    }

    public Value onTrue() {
        return operation.operand(1);
    }

    public Value onFalse() {
        return operation.operand(2);
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitSelect(this);
    }
}
