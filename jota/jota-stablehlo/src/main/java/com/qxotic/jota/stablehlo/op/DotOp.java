package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.attr.Attribute;
import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Value;
import java.util.Optional;

/** Typed view of {@code stablehlo.dot}. Generated from the op table; see {@link Op}. */
public record DotOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.dot";

    public DotOp {
        OpSupport.checkName(operation, NAME);
    }

    public Value lhs() {
        return operation.operand(0);
    }

    public Value rhs() {
        return operation.operand(1);
    }

    /** Attribute {@code precision_config}. */
    public Optional<Attribute> precisionConfig() {
        return operation.attribute("precision_config");
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitDot(this);
    }
}
