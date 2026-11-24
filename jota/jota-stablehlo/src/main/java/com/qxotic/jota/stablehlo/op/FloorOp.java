package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Value;

/** Typed view of {@code stablehlo.floor}. Generated from the op table; see {@link Op}. */
public record FloorOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.floor";

    public FloorOp {
        OpSupport.checkName(operation, NAME);
    }

    public Value operand() {
        return operation.operand(0);
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitFloor(this);
    }
}
