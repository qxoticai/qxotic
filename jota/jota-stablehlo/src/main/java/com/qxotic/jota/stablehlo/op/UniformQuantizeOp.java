package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Value;

/**
 * Typed view of {@code stablehlo.uniform_quantize}. Generated from the op table; see {@link Op}.
 */
public record UniformQuantizeOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.uniform_quantize";

    public UniformQuantizeOp {
        OpSupport.checkName(operation, NAME);
    }

    public Value operand() {
        return operation.operand(0);
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitUniformQuantize(this);
    }
}
