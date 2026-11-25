package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Value;
import java.util.List;

/** Typed view of {@code func.call}. Generated from the op table; see {@link Op}. */
public record CallOp(Operation operation) implements Op {

    public static final String NAME = "func.call";

    public CallOp {
        OpSupport.checkName(operation, NAME);
    }

    public List<Value> inputs() {
        return OpSupport.slice(operation, 0, 0);
    }

    /** Attribute {@code callee}. */
    public String callee() {
        return OpSupport.str(operation, "callee");
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitCall(this);
    }
}
