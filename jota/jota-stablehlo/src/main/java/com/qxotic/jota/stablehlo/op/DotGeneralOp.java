package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.attr.Attribute;
import com.qxotic.jota.stablehlo.attr.DotDimensionNumbers;
import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Value;
import java.util.Optional;

/** Typed view of {@code stablehlo.dot_general}. Generated from the op table; see {@link Op}. */
public record DotGeneralOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.dot_general";

    public DotGeneralOp {
        OpSupport.checkName(operation, NAME);
    }

    public Value lhs() {
        return operation.operand(0);
    }

    public Value rhs() {
        return operation.operand(1);
    }

    /** Attribute {@code dot_dimension_numbers}. */
    public DotDimensionNumbers dotDimensionNumbers() {
        return DotDimensionNumbers.from(OpSupport.required(operation, "dot_dimension_numbers"));
    }

    /** Attribute {@code precision_config}. */
    public Optional<Attribute> precisionConfig() {
        return operation.attribute("precision_config");
    }

    /** Attribute {@code algorithm}. */
    public Optional<Attribute> algorithm() {
        return operation.attribute("algorithm");
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitDotGeneral(this);
    }
}
