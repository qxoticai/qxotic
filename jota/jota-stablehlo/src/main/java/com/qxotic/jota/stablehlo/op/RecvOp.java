package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.attr.Attribute;
import com.qxotic.jota.stablehlo.attr.DenseElementsAttr;
import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Value;
import java.util.Optional;

/** Typed view of {@code stablehlo.recv}. Generated from the op table; see {@link Op}. */
public record RecvOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.recv";

    public RecvOp {
        OpSupport.checkName(operation, NAME);
    }

    public Value token() {
        return operation.operand(0);
    }

    /** Attribute {@code channel_handle}. */
    public Attribute channelHandle() {
        return OpSupport.required(operation, "channel_handle");
    }

    /** Attribute {@code is_host_transfer}. */
    public boolean isHostTransfer() {
        return OpSupport.bool(operation, "is_host_transfer");
    }

    /** Attribute {@code source_target_pairs}. */
    public Optional<DenseElementsAttr> sourceTargetPairs() {
        return OpSupport.denseOpt(operation, "source_target_pairs");
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitRecv(this);
    }
}
