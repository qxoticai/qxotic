package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Value;

/** Typed view of {@code stablehlo.slice}. Generated from the op table; see {@link Op}. */
public record SliceOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.slice";

    public SliceOp {
        OpSupport.checkName(operation, NAME);
    }

    public Value operand() {
        return operation.operand(0);
    }

    /** Attribute {@code start_indices}. */
    public long[] startIndices() {
        return OpSupport.i64s(operation, "start_indices");
    }

    /** Attribute {@code limit_indices}. */
    public long[] limitIndices() {
        return OpSupport.i64s(operation, "limit_indices");
    }

    /** Attribute {@code strides}. */
    public long[] strides() {
        return OpSupport.i64s(operation, "strides");
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitSlice(this);
    }
}
