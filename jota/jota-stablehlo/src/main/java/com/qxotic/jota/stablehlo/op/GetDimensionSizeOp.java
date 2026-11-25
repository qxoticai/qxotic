package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Value;

/**
 * Typed view of {@code stablehlo.get_dimension_size}. Generated from the op table; see {@link Op}.
 */
public record GetDimensionSizeOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.get_dimension_size";

    public GetDimensionSizeOp {
        OpSupport.checkName(operation, NAME);
    }

    public Value operand() {
        return operation.operand(0);
    }

    /** Attribute {@code dimension}. */
    public long dimension() {
        return OpSupport.i64(operation, "dimension");
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitGetDimensionSize(this);
    }
}
