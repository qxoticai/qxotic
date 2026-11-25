package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Value;
import java.util.List;

/** Typed view of {@code stablehlo.dynamic_slice}. Generated from the op table; see {@link Op}. */
public record DynamicSliceOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.dynamic_slice";

    public DynamicSliceOp {
        OpSupport.checkName(operation, NAME);
    }

    public Value operand() {
        return operation.operand(0);
    }

    public List<Value> startIndices() {
        return OpSupport.slice(operation, 1, 0);
    }

    /** Attribute {@code slice_sizes}. */
    public long[] sliceSizes() {
        return OpSupport.i64s(operation, "slice_sizes");
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitDynamicSlice(this);
    }
}
