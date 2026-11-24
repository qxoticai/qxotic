package com.qxotic.jota.stablehlo.attr;

/** StableHLO enum {@code #stablehlo<result_accuracy_mode ...>}. */
public enum ResultAccuracyMode implements StableHloEnum {
    DEFAULT,
    HIGHEST,
    TOLERANCE;

    public static final String KIND = "result_accuracy_mode";

    @Override
    public String kind() {
        return KIND;
    }

    public static ResultAccuracyMode from(Attribute attribute) {
        return StableHloEnum.decode(ResultAccuracyMode.class, KIND, attribute);
    }
}
