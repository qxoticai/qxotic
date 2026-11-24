package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Value;

/**
 * Typed view of {@code stablehlo.real_dynamic_slice}. Generated from the op table; see {@link Op}.
 */
public record RealDynamicSliceOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.real_dynamic_slice";

    public RealDynamicSliceOp {
        OpSupport.checkName(operation, NAME);
    }

    public Value operand() {
        return operation.operand(0);
    }

    public Value startIndices() {
        return operation.operand(1);
    }

    public Value limitIndices() {
        return operation.operand(2);
    }

    public Value strides() {
        return operation.operand(3);
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitRealDynamicSlice(this);
    }
}
