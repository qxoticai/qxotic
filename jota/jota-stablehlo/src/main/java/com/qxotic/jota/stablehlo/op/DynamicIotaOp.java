package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Value;

/** Typed view of {@code stablehlo.dynamic_iota}. Generated from the op table; see {@link Op}. */
public record DynamicIotaOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.dynamic_iota";

    public DynamicIotaOp {
        OpSupport.checkName(operation, NAME);
    }

    public Value outputShape() {
        return operation.operand(0);
    }

    /** Attribute {@code iota_dimension}. */
    public long iotaDimension() {
        return OpSupport.i64(operation, "iota_dimension");
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitDynamicIota(this);
    }
}
