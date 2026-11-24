package com.qxotic.jota.stablehlo.attr;

import java.util.Objects;

/** A {@code name = value} entry of an attribute dictionary. */
public record NamedAttribute(String name, Attribute value) {

    public NamedAttribute {
        Objects.requireNonNull(name, "name");
        Objects.requireNonNull(value, "value");
    }

    @Override
    public String toString() {
        if (value instanceof UnitAttr) {
            return AttrSyntax.key(name);
        }
        return AttrSyntax.key(name) + " = " + value;
    }
}
