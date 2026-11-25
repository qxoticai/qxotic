package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Value;
import java.util.List;

/** Typed view of {@code stablehlo.after_all}. Generated from the op table; see {@link Op}. */
public record AfterAllOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.after_all";

    public AfterAllOp {
        OpSupport.checkName(operation, NAME);
    }

    public List<Value> inputs() {
        return OpSupport.slice(operation, 0, 0);
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitAfterAll(this);
    }
}
