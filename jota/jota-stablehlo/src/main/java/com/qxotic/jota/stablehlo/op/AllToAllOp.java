package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.attr.Attribute;
import com.qxotic.jota.stablehlo.attr.DenseElementsAttr;
import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Value;
import java.util.List;
import java.util.Optional;

/** Typed view of {@code stablehlo.all_to_all}. Generated from the op table; see {@link Op}. */
public record AllToAllOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.all_to_all";

    public AllToAllOp {
        OpSupport.checkName(operation, NAME);
    }

    public List<Value> operands() {
        return OpSupport.slice(operation, 0, 0);
    }

    /** Attribute {@code split_dimension}. */
    public long splitDimension() {
        return OpSupport.i64(operation, "split_dimension");
    }

    /** Attribute {@code concat_dimension}. */
    public long concatDimension() {
        return OpSupport.i64(operation, "concat_dimension");
    }

    /** Attribute {@code split_count}. */
    public long splitCount() {
        return OpSupport.i64(operation, "split_count");
    }

    /** Attribute {@code replica_groups}. */
    public Optional<DenseElementsAttr> replicaGroups() {
        return OpSupport.denseOpt(operation, "replica_groups");
    }

    /** Attribute {@code channel_handle}. */
    public Optional<Attribute> channelHandle() {
        return operation.attribute("channel_handle");
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitAllToAll(this);
    }
}
