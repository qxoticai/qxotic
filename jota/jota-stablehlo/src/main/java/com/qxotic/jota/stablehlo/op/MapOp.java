package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Region;
import com.qxotic.jota.stablehlo.ir.Value;
import java.util.List;

/** Typed view of {@code stablehlo.map}. Generated from the op table; see {@link Op}. */
public record MapOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.map";

    public MapOp {
        OpSupport.checkName(operation, NAME);
    }

    public List<Value> inputs() {
        return OpSupport.slice(operation, 0, 0);
    }

    /** Attribute {@code dimensions}. */
    public long[] dimensions() {
        return OpSupport.i64s(operation, "dimensions");
    }

    public Region computation() {
        return operation.region(0);
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitMap(this);
    }
}
