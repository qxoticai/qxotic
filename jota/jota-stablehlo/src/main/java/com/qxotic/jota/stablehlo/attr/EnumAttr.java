package com.qxotic.jota.stablehlo.attr;

import java.util.Objects;

/**
 * Dialect enum attribute {@code #stablehlo<comparison_direction GT>}: {@code dialect="stablehlo"},
 * {@code kind="comparison_direction"}, {@code value="GT"}.
 */
public record EnumAttr(String dialect, String kind, String value) implements Attribute {

    public EnumAttr {
        Objects.requireNonNull(dialect, "dialect");
        Objects.requireNonNull(kind, "kind");
        Objects.requireNonNull(value, "value");
    }

    public static EnumAttr stablehlo(String kind, String value) {
        return new EnumAttr("stablehlo", kind, value);
    }

    @Override
    public String toString() {
        return "#" + dialect + "<" + kind + " " + value + ">";
    }
}
