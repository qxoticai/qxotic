package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.ir.Operation;

/** Typed view of {@code stablehlo.create_token}. Generated from the op table; see {@link Op}. */
public record CreateTokenOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.create_token";

    public CreateTokenOp {
        OpSupport.checkName(operation, NAME);
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitCreateToken(this);
    }
}
