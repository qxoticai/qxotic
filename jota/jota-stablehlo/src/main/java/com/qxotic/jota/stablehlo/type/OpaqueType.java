package com.qxotic.jota.stablehlo.type;

/**
 * A type of a dialect this model does not know ({@code !dialect.name<...>}, {@code memref<...>},
 * {@code none}...), kept verbatim so that files round-trip.
 */
public record OpaqueType(String text) implements Type {

    @Override
    public String mlir() {
        return text;
    }

    @Override
    public String toString() {
        return text;
    }
}
