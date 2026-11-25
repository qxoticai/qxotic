package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Value;

/**
 * Typed view of {@code stablehlo.broadcast_in_dim}. Generated from the op table; see {@link Op}.
 */
public record BroadcastInDimOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.broadcast_in_dim";

    public BroadcastInDimOp {
        OpSupport.checkName(operation, NAME);
    }

    public Value operand() {
        return operation.operand(0);
    }

    /** Attribute {@code broadcast_dimensions}. */
    public long[] broadcastDimensions() {
        return OpSupport.i64s(operation, "broadcast_dimensions");
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitBroadcastInDim(this);
    }
}
