package com.qxotic.jota.stablehlo.attr;

/** StableHLO enum {@code #stablehlo<rng_algorithm ...>}. */
public enum RngAlgorithm implements StableHloEnum {
    DEFAULT,
    THREE_FRY,
    PHILOX;

    public static final String KIND = "rng_algorithm";

    @Override
    public String kind() {
        return KIND;
    }

    public static RngAlgorithm from(Attribute attribute) {
        return StableHloEnum.decode(RngAlgorithm.class, KIND, attribute);
    }
}
