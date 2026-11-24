package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Value;

/**
 * Typed view of {@code stablehlo.get_tuple_element}. Generated from the op table; see {@link Op}.
 */
public record GetTupleElementOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.get_tuple_element";

    public GetTupleElementOp {
        OpSupport.checkName(operation, NAME);
    }

    public Value operand() {
        return operation.operand(0);
    }

    /** Attribute {@code index}. */
    public long index() {
        return OpSupport.i64(operation, "index");
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitGetTupleElement(this);
    }
}
