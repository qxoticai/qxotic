package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.attr.RngAlgorithm;
import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Value;

/**
 * Typed view of {@code stablehlo.rng_bit_generator}. Generated from the op table; see {@link Op}.
 */
public record RngBitGeneratorOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.rng_bit_generator";

    public RngBitGeneratorOp {
        OpSupport.checkName(operation, NAME);
    }

    public Value initialState() {
        return operation.operand(0);
    }

    /** Attribute {@code rng_algorithm}. */
    public RngAlgorithm rngAlgorithm() {
        return RngAlgorithm.from(OpSupport.required(operation, "rng_algorithm"));
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitRngBitGenerator(this);
    }
}
