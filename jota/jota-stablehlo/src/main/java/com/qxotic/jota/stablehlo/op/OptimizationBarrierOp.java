package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Value;
import java.util.List;

/**
 * Typed view of {@code stablehlo.optimization_barrier}. Generated from the op table; see {@link
 * Op}.
 */
public record OptimizationBarrierOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.optimization_barrier";

    public OptimizationBarrierOp {
        OpSupport.checkName(operation, NAME);
    }

    public List<Value> operand() {
        return OpSupport.slice(operation, 0, 0);
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitOptimizationBarrier(this);
    }
}
