package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Value;

/**
 * Typed view of {@code stablehlo.round_nearest_even}. Generated from the op table; see {@link Op}.
 */
public record RoundNearestEvenOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.round_nearest_even";

    public RoundNearestEvenOp {
        OpSupport.checkName(operation, NAME);
    }

    public Value operand() {
        return operation.operand(0);
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitRoundNearestEven(this);
    }
}
