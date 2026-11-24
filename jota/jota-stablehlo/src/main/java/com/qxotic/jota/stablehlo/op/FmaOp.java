package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Value;

/** Typed view of {@code stablehlo.fma}. Generated from the op table; see {@link Op}. */
public record FmaOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.fma";

    public FmaOp {
        OpSupport.checkName(operation, NAME);
    }

    public Value a() {
        return operation.operand(0);
    }

    public Value b() {
        return operation.operand(1);
    }

    public Value c() {
        return operation.operand(2);
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitFma(this);
    }
}
