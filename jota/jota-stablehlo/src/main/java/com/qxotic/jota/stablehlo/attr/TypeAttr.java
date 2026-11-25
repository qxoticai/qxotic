package com.qxotic.jota.stablehlo.attr;

import com.qxotic.jota.stablehlo.type.Type;
import java.util.Objects;

/** A type used as an attribute value (e.g. {@code function_type}). */
public record TypeAttr(Type type) implements Attribute {

    public TypeAttr {
        Objects.requireNonNull(type, "type");
    }

    @Override
    public String toString() {
        return type.mlir();
    }
}
