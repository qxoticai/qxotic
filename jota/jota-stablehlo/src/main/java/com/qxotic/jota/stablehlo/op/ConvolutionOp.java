package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.attr.Attribute;
import com.qxotic.jota.stablehlo.attr.ConvDimensionNumbers;
import com.qxotic.jota.stablehlo.attr.DenseElementsAttr;
import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Value;
import java.util.Optional;

/** Typed view of {@code stablehlo.convolution}. Generated from the op table; see {@link Op}. */
public record ConvolutionOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.convolution";

    public ConvolutionOp {
        OpSupport.checkName(operation, NAME);
    }

    public Value lhs() {
        return operation.operand(0);
    }

    public Value rhs() {
        return operation.operand(1);
    }

    /** Attribute {@code window_strides}. */
    public Optional<long[]> windowStrides() {
        return OpSupport.i64sOpt(operation, "window_strides");
    }

    /** Attribute {@code padding}. */
    public Optional<DenseElementsAttr> padding() {
        return OpSupport.denseOpt(operation, "padding");
    }

    /** Attribute {@code lhs_dilation}. */
    public Optional<long[]> lhsDilation() {
        return OpSupport.i64sOpt(operation, "lhs_dilation");
    }

    /** Attribute {@code rhs_dilation}. */
    public Optional<long[]> rhsDilation() {
        return OpSupport.i64sOpt(operation, "rhs_dilation");
    }

    /** Attribute {@code window_reversal}. */
    public Optional<Attribute> windowReversal() {
        return operation.attribute("window_reversal");
    }

    /** Attribute {@code dimension_numbers}. */
    public ConvDimensionNumbers dimensionNumbers() {
        return ConvDimensionNumbers.from(OpSupport.required(operation, "dimension_numbers"));
    }

    /** Attribute {@code feature_group_count}. */
    public long featureGroupCount() {
        return OpSupport.i64(operation, "feature_group_count");
    }

    /** Attribute {@code batch_group_count}. */
    public long batchGroupCount() {
        return OpSupport.i64(operation, "batch_group_count");
    }

    /** Attribute {@code precision_config}. */
    public Optional<Attribute> precisionConfig() {
        return operation.attribute("precision_config");
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitConvolution(this);
    }
}
