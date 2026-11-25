package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Value;

/** Typed view of {@code stablehlo.negate}. Generated from the op table; see {@link Op}. */
public record NegateOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.negate";

    public NegateOp {
        OpSupport.checkName(operation, NAME);
    }

    public Value operand() {
        return operation.operand(0);
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitNegate(this);
    }
}
