package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Value;

/** Typed view of {@code stablehlo.reverse}. Generated from the op table; see {@link Op}. */
public record ReverseOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.reverse";

    public ReverseOp {
        OpSupport.checkName(operation, NAME);
    }

    public Value operand() {
        return operation.operand(0);
    }

    /** Attribute {@code dimensions}. */
    public long[] dimensions() {
        return OpSupport.i64s(operation, "dimensions");
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitReverse(this);
    }
}
