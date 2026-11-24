package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.attr.Attribute;
import com.qxotic.jota.stablehlo.ir.Operation;

/** Typed view of {@code stablehlo.constant}. Generated from the op table; see {@link Op}. */
public record ConstantOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.constant";

    public ConstantOp {
        OpSupport.checkName(operation, NAME);
    }

    /** Attribute {@code value}. */
    public Attribute value() {
        return OpSupport.required(operation, "value");
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitConstant(this);
    }
}
