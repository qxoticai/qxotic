package com.qxotic.jota.stablehlo.attr;

/** StableHLO enum {@code #stablehlo<comparison_direction ...>}. */
public enum ComparisonDirection implements StableHloEnum {
    EQ,
    NE,
    GE,
    GT,
    LE,
    LT;

    public static final String KIND = "comparison_direction";

    @Override
    public String kind() {
        return KIND;
    }

    public static ComparisonDirection from(Attribute attribute) {
        return StableHloEnum.decode(ComparisonDirection.class, KIND, attribute);
    }
}
