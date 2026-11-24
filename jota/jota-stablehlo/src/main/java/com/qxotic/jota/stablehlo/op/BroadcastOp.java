package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Value;

/** Typed view of {@code stablehlo.broadcast}. Generated from the op table; see {@link Op}. */
public record BroadcastOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.broadcast";

    public BroadcastOp {
        OpSupport.checkName(operation, NAME);
    }

    public Value operand() {
        return operation.operand(0);
    }

    /** Attribute {@code broadcast_sizes}. */
    public long[] broadcastSizes() {
        return OpSupport.i64s(operation, "broadcast_sizes");
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitBroadcast(this);
    }
}
