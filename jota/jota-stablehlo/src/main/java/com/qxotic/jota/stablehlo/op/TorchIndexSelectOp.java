package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Value;

/**
 * Typed view of {@code stablehlo.torch_index_select}. Generated from the op table; see {@link Op}.
 */
public record TorchIndexSelectOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.torch_index_select";

    public TorchIndexSelectOp {
        OpSupport.checkName(operation, NAME);
    }

    public Value operand() {
        return operation.operand(0);
    }

    public Value index() {
        return operation.operand(1);
    }

    /** Attribute {@code dim}. */
    public long dim() {
        return OpSupport.i64(operation, "dim");
    }

    /** Attribute {@code batch_dims}. */
    public long batchDims() {
        return OpSupport.i64(operation, "batch_dims");
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitTorchIndexSelect(this);
    }
}
