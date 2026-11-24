package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Region;
import com.qxotic.jota.stablehlo.ir.Value;

/** Typed view of {@code stablehlo.if}. Generated from the op table; see {@link Op}. */
public record IfOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.if";

    public IfOp {
        OpSupport.checkName(operation, NAME);
    }

    public Value pred() {
        return operation.operand(0);
    }

    public Region trueBranch() {
        return operation.region(0);
    }

    public Region falseBranch() {
        return operation.region(1);
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitIf(this);
    }
}
