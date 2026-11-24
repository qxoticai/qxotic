package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.attr.Attribute;
import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Value;
import java.util.Optional;

/** Typed view of {@code stablehlo.logistic}. Generated from the op table; see {@link Op}. */
public record LogisticOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.logistic";

    public LogisticOp {
        OpSupport.checkName(operation, NAME);
    }

    public Value operand() {
        return operation.operand(0);
    }

    /** Attribute {@code result_accuracy}. */
    public Optional<Attribute> resultAccuracy() {
        return operation.attribute("result_accuracy");
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitLogistic(this);
    }
}
