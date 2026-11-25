package com.qxotic.jota.stablehlo.attr;

/** StableHLO enum {@code #stablehlo<precision ...>}. */
public enum Precision implements StableHloEnum {
    DEFAULT,
    HIGH,
    HIGHEST;

    public static final String KIND = "precision";

    @Override
    public String kind() {
        return KIND;
    }

    public static Precision from(Attribute attribute) {
        return StableHloEnum.decode(Precision.class, KIND, attribute);
    }
}
