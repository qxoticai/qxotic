package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.ir.Operation;

/** Typed view of {@code stablehlo.replica_id}. Generated from the op table; see {@link Op}. */
public record ReplicaIdOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.replica_id";

    public ReplicaIdOp {
        OpSupport.checkName(operation, NAME);
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitReplicaId(this);
    }
}
