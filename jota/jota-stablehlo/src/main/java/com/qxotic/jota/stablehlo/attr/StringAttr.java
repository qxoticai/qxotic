package com.qxotic.jota.stablehlo.attr;

import java.util.Objects;

/** String literal; each char holds one byte (0-255), the encoding is not interpreted. */
public record StringAttr(String value) implements Attribute {

    public StringAttr {
        Objects.requireNonNull(value, "value");
    }

    @Override
    public String toString() {
        return AttrSyntax.quote(value);
    }
}
