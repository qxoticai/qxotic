package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.attr.Attribute;
import com.qxotic.jota.stablehlo.attr.DenseElementsAttr;
import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Region;
import com.qxotic.jota.stablehlo.ir.Value;
import java.util.List;
import java.util.Optional;

/** Typed view of {@code stablehlo.all_reduce}. Generated from the op table; see {@link Op}. */
public record AllReduceOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.all_reduce";

    public AllReduceOp {
        OpSupport.checkName(operation, NAME);
    }

    public List<Value> operands() {
        return OpSupport.slice(operation, 0, 0);
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
        return visitor.visitAllReduce(this);
    }
}
