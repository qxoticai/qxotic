package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Value;

/** Typed view of {@code stablehlo.pad}. Generated from the op table; see {@link Op}. */
public record PadOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.pad";

    public PadOp {
        OpSupport.checkName(operation, NAME);
    }

    public Value operand() {
        return operation.operand(0);
    }

    public Value paddingValue() {
        return operation.operand(1);
    }

    /** Attribute {@code edge_padding_low}. */
    public long[] edgePaddingLow() {
        return OpSupport.i64s(operation, "edge_padding_low");
    }

    /** Attribute {@code edge_padding_high}. */
    public long[] edgePaddingHigh() {
        return OpSupport.i64s(operation, "edge_padding_high");
    }

    /** Attribute {@code interior_padding}. */
    public long[] interiorPadding() {
        return OpSupport.i64s(operation, "interior_padding");
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitPad(this);
    }
}
