package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Value;

/** Typed view of {@code stablehlo.async_done}. Generated from the op table; see {@link Op}. */
public record AsyncDoneOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.async_done";

    public AsyncDoneOp {
        OpSupport.checkName(operation, NAME);
    }

    public Value future() {
        return operation.operand(0);
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitAsyncDone(this);
    }
}
