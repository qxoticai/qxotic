package com.qxotic.jota.stablehlo.attr;

/** Presence-only attribute ({@code {foo}} or {@code unit}). */
public record UnitAttr() implements Attribute {

    public static final UnitAttr INSTANCE = new UnitAttr();

    @Override
    public String toString() {
        return "unit";
    }
}
