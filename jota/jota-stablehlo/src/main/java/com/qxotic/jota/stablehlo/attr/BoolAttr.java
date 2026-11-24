package com.qxotic.jota.stablehlo.attr;

/** {@code true} / {@code false}. */
public record BoolAttr(boolean value) implements Attribute {

    public static final BoolAttr TRUE = new BoolAttr(true);
    public static final BoolAttr FALSE = new BoolAttr(false);

    public static BoolAttr of(boolean value) {
        return value ? TRUE : FALSE;
    }

    @Override
    public String toString() {
        return Boolean.toString(value);
    }
}
