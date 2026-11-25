package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Value;
import java.util.List;
import java.util.Optional;

/** Typed view of {@code stablehlo.outfeed}. Generated from the op table; see {@link Op}. */
public record OutfeedOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.outfeed";

    public OutfeedOp {
        OpSupport.checkName(operation, NAME);
    }

    public List<Value> inputs() {
        return OpSupport.slice(operation, 0, 1);
    }

    public Value token() {
        return operation.operand(operation.numOperands() - 1);
    }

    /** Attribute {@code outfeed_config}. */
    public Optional<String> outfeedConfig() {
        return OpSupport.strOpt(operation, "outfeed_config");
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitOutfeed(this);
    }
}
