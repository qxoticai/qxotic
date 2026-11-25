package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Value;
import java.util.List;

/** Typed view of {@code stablehlo.concatenate}. Generated from the op table; see {@link Op}. */
public record ConcatenateOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.concatenate";

    public ConcatenateOp {
        OpSupport.checkName(operation, NAME);
    }

    public List<Value> inputs() {
        return OpSupport.slice(operation, 0, 0);
    }

    /** Attribute {@code dimension}. */
    public long dimension() {
        return OpSupport.i64(operation, "dimension");
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitConcatenate(this);
    }
}
