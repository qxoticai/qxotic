package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.attr.GatherDimensionNumbers;
import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Value;

/** Typed view of {@code stablehlo.dynamic_gather}. Generated from the op table; see {@link Op}. */
public record DynamicGatherOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.dynamic_gather";

    public DynamicGatherOp {
        OpSupport.checkName(operation, NAME);
    }

    public Value operand() {
        return operation.operand(0);
    }

    public Value startIndices() {
        return operation.operand(1);
    }

    public Value sliceSizes() {
        return operation.operand(2);
    }

    /** Attribute {@code dimension_numbers}. */
    public GatherDimensionNumbers dimensionNumbers() {
        return GatherDimensionNumbers.from(OpSupport.required(operation, "dimension_numbers"));
    }

    /** Attribute {@code indices_are_sorted}. */
    public boolean indicesAreSorted() {
        return OpSupport.bool(operation, "indices_are_sorted");
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitDynamicGather(this);
    }
}
