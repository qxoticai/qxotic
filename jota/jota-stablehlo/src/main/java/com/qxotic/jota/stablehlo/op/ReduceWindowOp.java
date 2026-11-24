package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.attr.DenseElementsAttr;
import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Region;
import com.qxotic.jota.stablehlo.ir.Value;
import java.util.List;
import java.util.Optional;

/** Typed view of {@code stablehlo.reduce_window}. Generated from the op table; see {@link Op}. */
public record ReduceWindowOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.reduce_window";

    public ReduceWindowOp {
        OpSupport.checkName(operation, NAME);
    }

    public List<Value> inputs() {
        return OpSupport.half(operation, 0);
    }

    public List<Value> initValues() {
        return OpSupport.half(operation, 1);
    }

    /** Attribute {@code window_dimensions}. */
    public long[] windowDimensions() {
        return OpSupport.i64s(operation, "window_dimensions");
    }

    /** Attribute {@code window_strides}. */
    public Optional<long[]> windowStrides() {
        return OpSupport.i64sOpt(operation, "window_strides");
    }

    /** Attribute {@code base_dilations}. */
    public Optional<long[]> baseDilations() {
        return OpSupport.i64sOpt(operation, "base_dilations");
    }

    /** Attribute {@code window_dilations}. */
    public Optional<long[]> windowDilations() {
        return OpSupport.i64sOpt(operation, "window_dilations");
    }

    /** Attribute {@code padding}. */
    public Optional<DenseElementsAttr> padding() {
        return OpSupport.denseOpt(operation, "padding");
    }

    public Region body() {
        return operation.region(0);
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitReduceWindow(this);
    }
}
