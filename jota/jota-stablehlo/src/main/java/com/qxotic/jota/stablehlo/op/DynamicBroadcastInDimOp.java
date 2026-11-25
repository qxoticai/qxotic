package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Value;
import java.util.Optional;

/**
 * Typed view of {@code stablehlo.dynamic_broadcast_in_dim}. Generated from the op table; see {@link
 * Op}.
 */
public record DynamicBroadcastInDimOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.dynamic_broadcast_in_dim";

    public DynamicBroadcastInDimOp {
        OpSupport.checkName(operation, NAME);
    }

    public Value operand() {
        return operation.operand(0);
    }

    public Value outputDimensions() {
        return operation.operand(1);
    }

    /** Attribute {@code broadcast_dimensions}. */
    public long[] broadcastDimensions() {
        return OpSupport.i64s(operation, "broadcast_dimensions");
    }

    /** Attribute {@code known_expanding_dimensions}. */
    public Optional<long[]> knownExpandingDimensions() {
        return OpSupport.i64sOpt(operation, "known_expanding_dimensions");
    }

    /** Attribute {@code known_nonexpanding_dimensions}. */
    public Optional<long[]> knownNonexpandingDimensions() {
        return OpSupport.i64sOpt(operation, "known_nonexpanding_dimensions");
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitDynamicBroadcastInDim(this);
    }
}
