package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.attr.GatherDimensionNumbers;
import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Value;

/** Typed view of {@code stablehlo.gather}. Generated from the op table; see {@link Op}. */
public record GatherOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.gather";

    public GatherOp {
        OpSupport.checkName(operation, NAME);
    }

    public Value operand() {
        return operation.operand(0);
    }

    public Value startIndices() {
        return operation.operand(1);
    }

    /** Attribute {@code dimension_numbers}. */
    public GatherDimensionNumbers dimensionNumbers() {
        return GatherDimensionNumbers.from(OpSupport.required(operation, "dimension_numbers"));
    }

    /** Attribute {@code slice_sizes}. */
    public long[] sliceSizes() {
        return OpSupport.i64s(operation, "slice_sizes");
    }

    /** Attribute {@code indices_are_sorted}. */
    public boolean indicesAreSorted() {
        return OpSupport.bool(operation, "indices_are_sorted");
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitGather(this);
    }
}
