package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.attr.Attribute;
import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Value;
import java.util.List;
import java.util.Optional;

/** Typed view of {@code stablehlo.async_start}. Generated from the op table; see {@link Op}. */
public record AsyncStartOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.async_start";

    public AsyncStartOp {
        OpSupport.checkName(operation, NAME);
    }

    public List<Value> inputs() {
        return OpSupport.slice(operation, 0, 0);
    }

    /** Attribute {@code called_computation}. */
    public Optional<Attribute> calledComputation() {
        return operation.attribute("called_computation");
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitAsyncStart(this);
    }
}
