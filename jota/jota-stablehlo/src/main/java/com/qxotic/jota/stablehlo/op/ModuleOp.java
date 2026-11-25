package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Region;
import java.util.Optional;

/** Typed view of {@code builtin.module}. Generated from the op table; see {@link Op}. */
public record ModuleOp(Operation operation) implements Op {

    public static final String NAME = "builtin.module";

    public ModuleOp {
        OpSupport.checkName(operation, NAME);
    }

    /** Attribute {@code sym_name}. */
    public Optional<String> symName() {
        return OpSupport.strOpt(operation, "sym_name");
    }

    public Region bodyRegion() {
        return operation.region(0);
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitModule(this);
    }
}
