package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Value;
import java.util.List;

/**
 * Typed view of {@code stablehlo.dynamic_update_slice}. Generated from the op table; see {@link
 * Op}.
 */
public record DynamicUpdateSliceOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.dynamic_update_slice";

    public DynamicUpdateSliceOp {
        OpSupport.checkName(operation, NAME);
    }

    public Value operand() {
        return operation.operand(0);
    }

    public Value update() {
        return operation.operand(1);
    }

    public List<Value> startIndices() {
        return OpSupport.slice(operation, 2, 0);
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitDynamicUpdateSlice(this);
    }
}
