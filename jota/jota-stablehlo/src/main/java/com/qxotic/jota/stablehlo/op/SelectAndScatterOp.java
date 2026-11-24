package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.attr.DenseElementsAttr;
import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Region;
import com.qxotic.jota.stablehlo.ir.Value;
import java.util.Optional;

/**
 * Typed view of {@code stablehlo.select_and_scatter}. Generated from the op table; see {@link Op}.
 */
public record SelectAndScatterOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.select_and_scatter";

    public SelectAndScatterOp {
        OpSupport.checkName(operation, NAME);
    }

    public Value operand() {
        return operation.operand(0);
    }

    public Value source() {
        return operation.operand(1);
    }

    public Value initValue() {
        return operation.operand(2);
    }

    /** Attribute {@code window_dimensions}. */
    public Optional<long[]> windowDimensions() {
        return OpSupport.i64sOpt(operation, "window_dimensions");
    }

    /** Attribute {@code window_strides}. */
    public Optional<long[]> windowStrides() {
        return OpSupport.i64sOpt(operation, "window_strides");
    }

    /** Attribute {@code padding}. */
    public Optional<DenseElementsAttr> padding() {
        return OpSupport.denseOpt(operation, "padding");
    }

    public Region select() {
        return operation.region(0);
    }

    public Region scatter() {
        return operation.region(1);
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitSelectAndScatter(this);
    }
}
