package com.qxotic.jota.stablehlo.attr;

/** StableHLO enum {@code #stablehlo<rng_distribution ...>}. */
public enum RngDistribution implements StableHloEnum {
    UNIFORM,
    NORMAL;

    public static final String KIND = "rng_distribution";

    @Override
    public String kind() {
        return KIND;
    }

    public static RngDistribution from(Attribute attribute) {
        return StableHloEnum.decode(RngDistribution.class, KIND, attribute);
    }
}
