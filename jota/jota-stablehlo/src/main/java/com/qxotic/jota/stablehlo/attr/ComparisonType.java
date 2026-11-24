package com.qxotic.jota.stablehlo.attr;

/** StableHLO enum {@code #stablehlo<comparison_type ...>}. */
public enum ComparisonType implements StableHloEnum {
    NOTYPE,
    FLOAT,
    TOTALORDER,
    SIGNED,
    UNSIGNED,
    /** Since StableHLO 1.22.0; floats only. */
    WEAKORDER;

    public static final String KIND = "comparison_type";

    @Override
    public String kind() {
        return KIND;
    }

    public static ComparisonType from(Attribute attribute) {
        return StableHloEnum.decode(ComparisonType.class, KIND, attribute);
    }
}
