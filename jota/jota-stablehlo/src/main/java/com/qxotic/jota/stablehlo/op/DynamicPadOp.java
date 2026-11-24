package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Value;

/** Typed view of {@code stablehlo.dynamic_pad}. Generated from the op table; see {@link Op}. */
public record DynamicPadOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.dynamic_pad";

    public DynamicPadOp {
        OpSupport.checkName(operation, NAME);
    }

    public Value operand() {
        return operation.operand(0);
    }

    public Value paddingValue() {
        return operation.operand(1);
    }

    public Value edgePaddingLow() {
        return operation.operand(2);
    }

    public Value edgePaddingHigh() {
        return operation.operand(3);
    }

    public Value interiorPadding() {
        return operation.operand(4);
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitDynamicPad(this);
    }
}
