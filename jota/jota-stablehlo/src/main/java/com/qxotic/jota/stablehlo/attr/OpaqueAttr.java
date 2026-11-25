package com.qxotic.jota.stablehlo.attr;

/**
 * An attribute this model does not interpret ({@code affine_map<...>}, {@code dense_resource<...>},
 * {@code #sdy.sharding<...>} with a non key/value body...), kept verbatim so that files round-trip.
 */
public record OpaqueAttr(String text) implements Attribute {

    @Override
    public String toString() {
        return text;
    }
}
