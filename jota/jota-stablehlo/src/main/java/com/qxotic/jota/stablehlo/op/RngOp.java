package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.attr.RngDistribution;
import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Value;

/** Typed view of {@code stablehlo.rng}. Generated from the op table; see {@link Op}. */
public record RngOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.rng";

    public RngOp {
        OpSupport.checkName(operation, NAME);
    }

    public Value a() {
        return operation.operand(0);
    }

    public Value b() {
        return operation.operand(1);
    }

    public Value shape() {
        return operation.operand(2);
    }

    /** Attribute {@code rng_distribution}. */
    public RngDistribution rngDistribution() {
        return RngDistribution.from(OpSupport.required(operation, "rng_distribution"));
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitRng(this);
    }
}
