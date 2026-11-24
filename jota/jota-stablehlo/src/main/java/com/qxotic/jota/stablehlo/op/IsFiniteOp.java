package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Value;

/** Typed view of {@code stablehlo.is_finite}. Generated from the op table; see {@link Op}. */
public record IsFiniteOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.is_finite";

    public IsFiniteOp {
        OpSupport.checkName(operation, NAME);
    }

    public Value operand() {
        return operation.operand(0);
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitIsFinite(this);
    }
}
