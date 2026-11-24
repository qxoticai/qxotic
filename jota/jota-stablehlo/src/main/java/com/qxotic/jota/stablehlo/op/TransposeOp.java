package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Value;

/** Typed view of {@code stablehlo.transpose}. Generated from the op table; see {@link Op}. */
public record TransposeOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.transpose";

    public TransposeOp {
        OpSupport.checkName(operation, NAME);
    }

    public Value operand() {
        return operation.operand(0);
    }

    /** Attribute {@code permutation}. */
    public long[] permutation() {
        return OpSupport.i64s(operation, "permutation");
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitTranspose(this);
    }
}
