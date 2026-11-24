package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.attr.Attribute;
import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Value;
import java.util.Optional;

/** Typed view of {@code stablehlo.infeed}. Generated from the op table; see {@link Op}. */
public record InfeedOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.infeed";

    public InfeedOp {
        OpSupport.checkName(operation, NAME);
    }

    public Value token() {
        return operation.operand(0);
    }

    /** Attribute {@code infeed_config}. */
    public Optional<String> infeedConfig() {
        return OpSupport.strOpt(operation, "infeed_config");
    }

    /** Attribute {@code layout}. */
    public Optional<Attribute> layout() {
        return operation.attribute("layout");
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitInfeed(this);
    }
}
