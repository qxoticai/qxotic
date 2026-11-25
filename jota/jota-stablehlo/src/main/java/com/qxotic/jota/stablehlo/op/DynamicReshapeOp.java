package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Value;

/** Typed view of {@code stablehlo.dynamic_reshape}. Generated from the op table; see {@link Op}. */
public record DynamicReshapeOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.dynamic_reshape";

    public DynamicReshapeOp {
        OpSupport.checkName(operation, NAME);
    }

    public Value operand() {
        return operation.operand(0);
    }

    public Value outputShape() {
        return operation.operand(1);
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitDynamicReshape(this);
    }
}
