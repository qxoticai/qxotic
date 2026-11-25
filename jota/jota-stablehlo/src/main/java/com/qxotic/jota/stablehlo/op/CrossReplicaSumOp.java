package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.attr.DenseElementsAttr;
import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Value;

/**
 * Typed view of {@code stablehlo.cross-replica-sum} (legacy all_reduce). Generated from the op
 * table; see {@link Op}.
 */
public record CrossReplicaSumOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.cross-replica-sum";

    public CrossReplicaSumOp {
        OpSupport.checkName(operation, NAME);
    }

    public Value operand() {
        return operation.operand(0);
    }

    /** Attribute {@code replica_groups}. */
    public DenseElementsAttr replicaGroups() {
        return OpSupport.dense(operation, "replica_groups");
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitCrossReplicaSum(this);
    }
}
