package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Value;

/** Typed view of {@code stablehlo.multiply}. Generated from the op table; see {@link Op}. */
public record MultiplyOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.multiply";

    public MultiplyOp {
        OpSupport.checkName(operation, NAME);
    }

    public Value lhs() {
        return operation.operand(0);
    }

    public Value rhs() {
        return operation.operand(1);
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitMultiply(this);
    }
}
