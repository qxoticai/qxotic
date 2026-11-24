package com.qxotic.jota.stablehlo.attr;

/** Common behaviour of the StableHLO enum attributes. */
public interface StableHloEnum {

    /** The enum kind as spelled in {@code #stablehlo<kind VALUE>}. */
    String kind();

    String name();

    default EnumAttr toAttribute() {
        return EnumAttr.stablehlo(kind(), name());
    }

    /**
     * Decodes {@code #stablehlo<kind VALUE>} or {@code #stablehlo.kind<VALUE>} (the form used for
     * enums nested in other attributes).
     */
    static <E extends Enum<E> & StableHloEnum> E decode(
            Class<E> type, String kind, Attribute attribute) {
        String value =
                switch (attribute) {
                    case EnumAttr e when e.kind().equals(kind) -> e.value();
                    case DialectAttr d when d.mnemonic().equals(kind) && !d.hasParams() ->
                            d.rawBody().trim();
                    default ->
                            throw new IllegalArgumentException(
                                    "expected #stablehlo<" + kind + " ...>, got " + attribute);
                };
        return Enum.valueOf(type, value);
    }
}
