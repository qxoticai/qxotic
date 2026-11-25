package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Value;

/** Typed view of {@code stablehlo.cholesky}. Generated from the op table; see {@link Op}. */
public record CholeskyOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.cholesky";

    public CholeskyOp {
        OpSupport.checkName(operation, NAME);
    }

    public Value a() {
        return operation.operand(0);
    }

    /** Attribute {@code lower}. */
    public boolean lower() {
        return OpSupport.bool(operation, "lower");
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitCholesky(this);
    }
}
