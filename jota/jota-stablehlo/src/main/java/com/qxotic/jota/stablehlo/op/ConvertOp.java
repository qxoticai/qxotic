package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Value;

/** Typed view of {@code stablehlo.convert}. Generated from the op table; see {@link Op}. */
public record ConvertOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.convert";

    public ConvertOp {
        OpSupport.checkName(operation, NAME);
    }

    public Value operand() {
        return operation.operand(0);
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitConvert(this);
    }
}
