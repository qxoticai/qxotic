package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Value;

/** Typed view of {@code stablehlo.unary_einsum}. Generated from the op table; see {@link Op}. */
public record UnaryEinsumOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.unary_einsum";

    public UnaryEinsumOp {
        OpSupport.checkName(operation, NAME);
    }

    public Value operand() {
        return operation.operand(0);
    }

    /** Attribute {@code einsum_config}. */
    public String einsumConfig() {
        return OpSupport.str(operation, "einsum_config");
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitUnaryEinsum(this);
    }
}
