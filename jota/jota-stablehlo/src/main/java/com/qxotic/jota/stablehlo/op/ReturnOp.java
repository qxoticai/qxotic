package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Value;
import java.util.List;

/**
 * Typed view of {@code stablehlo.return}: terminator of StableHLO regions. Generated from the op
 * table; see {@link Op}.
 */
public record ReturnOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.return";

    public ReturnOp {
        OpSupport.checkName(operation, NAME);
    }

    public List<Value> results() {
        return OpSupport.slice(operation, 0, 0);
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitReturn(this);
    }
}
