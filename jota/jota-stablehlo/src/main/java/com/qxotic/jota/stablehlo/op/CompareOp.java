package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.attr.ComparisonDirection;
import com.qxotic.jota.stablehlo.attr.ComparisonType;
import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Value;
import java.util.Optional;

/** Typed view of {@code stablehlo.compare}. Generated from the op table; see {@link Op}. */
public record CompareOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.compare";

    public CompareOp {
        OpSupport.checkName(operation, NAME);
    }

    public Value lhs() {
        return operation.operand(0);
    }

    public Value rhs() {
        return operation.operand(1);
    }

    /** Attribute {@code comparison_direction}. */
    public ComparisonDirection comparisonDirection() {
        return ComparisonDirection.from(OpSupport.required(operation, "comparison_direction"));
    }

    /** Attribute {@code compare_type}. */
    public Optional<ComparisonType> compareType() {
        return operation.attribute("compare_type").map(ComparisonType::from);
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitCompare(this);
    }
}
