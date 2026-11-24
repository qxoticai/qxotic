package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.attr.Attribute;
import com.qxotic.jota.stablehlo.attr.DenseElementsAttr;
import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Value;
import java.util.List;
import java.util.Optional;

/** Typed view of {@code stablehlo.send}. Generated from the op table; see {@link Op}. */
public record SendOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.send";

    public SendOp {
        OpSupport.checkName(operation, NAME);
    }

    public List<Value> inputs() {
        return OpSupport.slice(operation, 0, 1);
    }

    public Value token() {
        return operation.operand(operation.numOperands() - 1);
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
        return visitor.visitSend(this);
    }
}
