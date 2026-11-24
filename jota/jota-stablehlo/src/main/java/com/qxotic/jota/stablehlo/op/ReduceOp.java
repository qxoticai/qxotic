package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Region;
import com.qxotic.jota.stablehlo.ir.Value;
import java.util.List;

/** Typed view of {@code stablehlo.reduce}. Generated from the op table; see {@link Op}. */
public record ReduceOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.reduce";

    public ReduceOp {
        OpSupport.checkName(operation, NAME);
    }

    public List<Value> inputs() {
        return OpSupport.half(operation, 0);
    }

    public List<Value> initValues() {
        return OpSupport.half(operation, 1);
    }

    /** Attribute {@code dimensions}. */
    public long[] dimensions() {
        return OpSupport.i64s(operation, "dimensions");
    }

    public Region body() {
        return operation.region(0);
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitReduce(this);
    }
}
