package com.qxotic.jota.stablehlo.attr;

/** StableHLO enum {@code #stablehlo<transpose ...>}. */
public enum Transpose implements StableHloEnum {
    TRANSPOSE_INVALID,
    NO_TRANSPOSE,
    TRANSPOSE,
    ADJOINT;

    public static final String KIND = "transpose";

    @Override
    public String kind() {
        return KIND;
    }

    public static Transpose from(Attribute attribute) {
        return StableHloEnum.decode(Transpose.class, KIND, attribute);
    }
}
