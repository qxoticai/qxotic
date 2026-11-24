package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Region;
import com.qxotic.jota.stablehlo.ir.Value;
import java.util.List;
import java.util.OptionalLong;

/** Typed view of {@code stablehlo.sort}. Generated from the op table; see {@link Op}. */
public record SortOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.sort";

    public SortOp {
        OpSupport.checkName(operation, NAME);
    }

    public List<Value> inputs() {
        return OpSupport.slice(operation, 0, 0);
    }

    /** Attribute {@code dimension}. */
    public OptionalLong dimension() {
        return OpSupport.i64Opt(operation, "dimension");
    }

    /** Attribute {@code is_stable}. */
    public boolean isStable() {
        return OpSupport.bool(operation, "is_stable");
    }

    public Region comparator() {
        return operation.region(0);
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitSort(this);
    }
}
