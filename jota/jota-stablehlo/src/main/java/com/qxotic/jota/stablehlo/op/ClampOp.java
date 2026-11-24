package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Value;

/** Typed view of {@code stablehlo.clamp}. Generated from the op table; see {@link Op}. */
public record ClampOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.clamp";

    public ClampOp {
        OpSupport.checkName(operation, NAME);
    }

    public Value min() {
        return operation.operand(0);
    }

    public Value operand() {
        return operation.operand(1);
    }

    public Value max() {
        return operation.operand(2);
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitClamp(this);
    }
}
