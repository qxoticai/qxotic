package com.qxotic.jota.stablehlo.type;

import java.util.List;
import java.util.stream.Collectors;

/** {@code tuple<T0, T1, ...>} (legacy HLO compatibility). */
public record TupleType(List<Type> types) implements Type {

    public TupleType {
        types = List.copyOf(types);
    }

    @Override
    public String mlir() {
        return types.stream().map(Type::mlir).collect(Collectors.joining(", ", "tuple<", ">"));
    }

    @Override
    public String toString() {
        return mlir();
    }
}
