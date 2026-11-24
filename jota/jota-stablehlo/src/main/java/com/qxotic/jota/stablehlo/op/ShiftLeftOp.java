package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Value;

/** Typed view of {@code stablehlo.shift_left}. Generated from the op table; see {@link Op}. */
public record ShiftLeftOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.shift_left";

    public ShiftLeftOp {
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
        return visitor.visitShiftLeft(this);
    }
}
