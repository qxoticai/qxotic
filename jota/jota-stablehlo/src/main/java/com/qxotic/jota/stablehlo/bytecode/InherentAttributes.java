package com.qxotic.jota.stablehlo.bytecode;

import java.util.List;
import java.util.Set;
import java.util.TreeSet;

/**
 * The inherent attributes of the registered ops (MLIR properties, {@code <{...}>} in the generic
 * form): those of every known property layout of the op, from the same tables the bytecode reader
 * and writer use.
 */
public final class InherentAttributes {

    private InherentAttributes() {}

    /**
     * The inherent attribute names of {@code opName} ({@code stablehlo.gather}, {@code chlo.top_k},
     * {@code func.func}...), or null when the op is not known.
     */
    public static Set<String> of(String opName) {
        List<List<PropertySchemas.Field>> layouts = PropertySchemas.layouts(opName);
        if (layouts == null) {
            return null;
        }
        Set<String> names = new TreeSet<>();
        for (List<PropertySchemas.Field> layout : layouts) {
            for (PropertySchemas.Field f : layout) {
                if (!f.name().equals(PropertySchemas.UNKNOWN)) {
                    names.add(f.name());
                }
            }
        }
        return names;
    }
}
