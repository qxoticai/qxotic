package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Value;

/**
 * Typed view of {@code stablehlo.batch_norm_inference}. Generated from the op table; see {@link
 * Op}.
 */
public record BatchNormInferenceOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.batch_norm_inference";

    public BatchNormInferenceOp {
        OpSupport.checkName(operation, NAME);
    }

    public Value operand() {
        return operation.operand(0);
    }

    public Value scale() {
        return operation.operand(1);
    }

    public Value offset() {
        return operation.operand(2);
    }

    public Value mean() {
        return operation.operand(3);
    }

    public Value variance() {
        return operation.operand(4);
    }

    /** Attribute {@code epsilon}. */
    public double epsilon() {
        return OpSupport.f64(operation, "epsilon");
    }

    /** Attribute {@code feature_index}. */
    public long featureIndex() {
        return OpSupport.i64(operation, "feature_index");
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitBatchNormInference(this);
    }
}
