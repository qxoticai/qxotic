package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.attr.Attribute;
import com.qxotic.jota.stablehlo.attr.DenseElementsAttr;
import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Value;
import java.util.Optional;

/**
 * Typed view of {@code stablehlo.collective_permute}. Generated from the op table; see {@link Op}.
 */
public record CollectivePermuteOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.collective_permute";

    public CollectivePermuteOp {
        OpSupport.checkName(operation, NAME);
    }

    public Value operand() {
        return operation.operand(0);
    }

    /** Attribute {@code source_target_pairs}. */
    public DenseElementsAttr sourceTargetPairs() {
        return OpSupport.dense(operation, "source_target_pairs");
    }

    /** Attribute {@code channel_handle}. */
    public Optional<Attribute> channelHandle() {
        return operation.attribute("channel_handle");
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitCollectivePermute(this);
    }
}
