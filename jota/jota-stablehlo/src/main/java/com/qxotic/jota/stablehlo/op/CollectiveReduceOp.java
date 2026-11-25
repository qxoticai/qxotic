package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.attr.Attribute;
import com.qxotic.jota.stablehlo.attr.DenseElementsAttr;
import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Region;
import com.qxotic.jota.stablehlo.ir.Value;
import java.util.Optional;

/**
 * Typed view of {@code stablehlo.collective_reduce}. Generated from the op table; see {@link Op}.
 */
public record CollectiveReduceOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.collective_reduce";

    public CollectiveReduceOp {
        OpSupport.checkName(operation, NAME);
    }

    public Value operand() {
        return operation.operand(0);
    }

    /** Attribute {@code replica_groups}. */
    public Optional<DenseElementsAttr> replicaGroups() {
        return OpSupport.denseOpt(operation, "replica_groups");
    }

    /** Attribute {@code channel_handle}. */
    public Optional<Attribute> channelHandle() {
        return operation.attribute("channel_handle");
    }

    public Region computation() {
        return operation.region(0);
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitCollectiveReduce(this);
    }
}
