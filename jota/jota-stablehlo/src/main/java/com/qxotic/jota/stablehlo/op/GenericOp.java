package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.ir.Operation;

/**
 * An operation without a typed record: other dialects ({@code chlo.*}, {@code check.*}, {@code
 * sdy.*}...) or unknown StableHLO ops. All data stays reachable through {@link #operation()}.
 */
public record GenericOp(Operation operation) implements Op {

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitGeneric(this);
    }
}
