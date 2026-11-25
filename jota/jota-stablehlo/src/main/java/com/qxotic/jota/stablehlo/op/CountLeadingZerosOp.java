package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Value;

/**
 * Typed view of {@code stablehlo.count_leading_zeros}. Generated from the op table; see {@link Op}.
 */
public record CountLeadingZerosOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.count_leading_zeros";

    public CountLeadingZerosOp {
        OpSupport.checkName(operation, NAME);
    }

    public Value operand() {
        return operation.operand(0);
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitCountLeadingZeros(this);
    }
}
