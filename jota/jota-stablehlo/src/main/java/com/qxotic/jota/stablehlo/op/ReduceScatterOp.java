package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.attr.Attribute;
import com.qxotic.jota.stablehlo.attr.DenseElementsAttr;
import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Region;
import com.qxotic.jota.stablehlo.ir.Value;
import java.util.Optional;

/** Typed view of {@code stablehlo.reduce_scatter}. Generated from the op table; see {@link Op}. */
public record ReduceScatterOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.reduce_scatter";

    public ReduceScatterOp {
        OpSupport.checkName(operation, NAME);
    }

    public Value operand() {
        return operation.operand(0);
    }

    /** Attribute {@code scatter_dimension}. */
    public long scatterDimension() {
        return OpSupport.i64(operation, "scatter_dimension");
    }

    /** Attribute {@code replica_groups}. */
    public Optional<DenseElementsAttr> replicaGroups() {
        return OpSupport.denseOpt(operation, "replica_groups");
    }

    /** Attribute {@code channel_handle}. */
    public Optional<Attribute> channelHandle() {
        return operation.attribute("channel_handle");
    }

    /** Attribute {@code use_global_device_ids}. */
    public boolean useGlobalDeviceIds() {
        return OpSupport.bool(operation, "use_global_device_ids");
    }

    public Region computation() {
        return operation.region(0);
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitReduceScatter(this);
    }
}
