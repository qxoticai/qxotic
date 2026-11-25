package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.attr.Attribute;
import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Value;
import java.util.List;
import java.util.Optional;
import java.util.OptionalLong;

/** Typed view of {@code stablehlo.composite}. Generated from the op table; see {@link Op}. */
public record CompositeOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.composite";

    public CompositeOp {
        OpSupport.checkName(operation, NAME);
    }

    public List<Value> inputs() {
        return OpSupport.slice(operation, 0, 0);
    }

    /** Attribute {@code name}. */
    public String compositeName() {
        return OpSupport.str(operation, "name");
    }

    /** Attribute {@code composite_attributes}. */
    public Optional<Attribute> compositeAttributes() {
        return operation.attribute("composite_attributes");
    }

    /** Attribute {@code decomposition}. */
    public String decomposition() {
        return OpSupport.str(operation, "decomposition");
    }

    /** Attribute {@code version}. */
    public OptionalLong version() {
        return OpSupport.i64Opt(operation, "version");
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitComposite(this);
    }
}
