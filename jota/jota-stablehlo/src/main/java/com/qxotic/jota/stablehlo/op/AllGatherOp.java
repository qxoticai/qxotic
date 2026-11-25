package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.attr.Attribute;
import com.qxotic.jota.stablehlo.attr.DenseElementsAttr;
import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Value;
import java.util.List;
import java.util.Optional;

/** Typed view of {@code stablehlo.all_gather}. Generated from the op table; see {@link Op}. */
public record AllGatherOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.all_gather";

    public AllGatherOp {
        OpSupport.checkName(operation, NAME);
    }

    public List<Value> operands() {
        return OpSupport.slice(operation, 0, 0);
    }

    /** Attribute {@code all_gather_dim}. */
    public long allGatherDim() {
        return OpSupport.i64(operation, "all_gather_dim");
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

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitAllGather(this);
    }
}
