package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Value;

/**
 * Typed view of {@code stablehlo.reduce_precision}. Generated from the op table; see {@link Op}.
 */
public record ReducePrecisionOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.reduce_precision";

    public ReducePrecisionOp {
        OpSupport.checkName(operation, NAME);
    }

    public Value operand() {
        return operation.operand(0);
    }

    /** Attribute {@code exponent_bits}. */
    public long exponentBits() {
        return OpSupport.i64(operation, "exponent_bits");
    }

    /** Attribute {@code mantissa_bits}. */
    public long mantissaBits() {
        return OpSupport.i64(operation, "mantissa_bits");
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitReducePrecision(this);
    }
}
