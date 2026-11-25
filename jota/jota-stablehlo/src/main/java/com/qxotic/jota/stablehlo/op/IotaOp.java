package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.ir.Operation;

/** Typed view of {@code stablehlo.iota}. Generated from the op table; see {@link Op}. */
public record IotaOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.iota";

    public IotaOp {
        OpSupport.checkName(operation, NAME);
    }

    /** Attribute {@code iota_dimension}. */
    public long iotaDimension() {
        return OpSupport.i64(operation, "iota_dimension");
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitIota(this);
    }
}
