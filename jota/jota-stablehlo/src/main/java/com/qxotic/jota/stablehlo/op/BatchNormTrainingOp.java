package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Value;

/**
 * Typed view of {@code stablehlo.batch_norm_training}. Generated from the op table; see {@link Op}.
 */
public record BatchNormTrainingOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.batch_norm_training";

    public BatchNormTrainingOp {
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
        return visitor.visitBatchNormTraining(this);
    }
}
