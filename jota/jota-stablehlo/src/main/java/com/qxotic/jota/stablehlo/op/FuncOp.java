package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.attr.Attribute;
import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Region;
import com.qxotic.jota.stablehlo.type.Type;
import java.util.Optional;

/**
 * Typed view of {@code func.func}: a named function. Generated from the op table; see {@link Op}.
 */
public record FuncOp(Operation operation) implements Op {

    public static final String NAME = "func.func";

    public FuncOp {
        OpSupport.checkName(operation, NAME);
    }

    /** Attribute {@code sym_name}. */
    public String symName() {
        return OpSupport.str(operation, "sym_name");
    }

    /** Attribute {@code function_type}. */
    public Type functionType() {
        return OpSupport.type(operation, "function_type");
    }

    /** Attribute {@code sym_visibility}. */
    public Optional<String> symVisibility() {
        return OpSupport.strOpt(operation, "sym_visibility");
    }

    /** Attribute {@code arg_attrs}. */
    public Optional<Attribute> argAttrs() {
        return operation.attribute("arg_attrs");
    }

    /** Attribute {@code res_attrs}. */
    public Optional<Attribute> resAttrs() {
        return operation.attribute("res_attrs");
    }

    public Region body() {
        return operation.region(0);
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitFunc(this);
    }
}
