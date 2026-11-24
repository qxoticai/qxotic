package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Value;

/** Typed view of {@code stablehlo.einsum}. Generated from the op table; see {@link Op}. */
public record EinsumOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.einsum";

    public EinsumOp {
        OpSupport.checkName(operation, NAME);
    }

    public Value lhs() {
        return operation.operand(0);
    }

    public Value rhs() {
        return operation.operand(1);
    }

    /** Attribute {@code einsum_config}. */
    public String einsumConfig() {
        return OpSupport.str(operation, "einsum_config");
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitEinsum(this);
    }
}
