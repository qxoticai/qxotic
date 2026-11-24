package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.ir.Operation;

/** Typed view of {@code stablehlo.partition_id}. Generated from the op table; see {@link Op}. */
public record PartitionIdOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.partition_id";

    public PartitionIdOp {
        OpSupport.checkName(operation, NAME);
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitPartitionId(this);
    }
}
