package com.qxotic.jota.stablehlo.type;

import java.util.List;
import java.util.stream.Collectors;

/** {@code !stablehlo.future<T0, ...>}: value produced by {@code async_start}. */
public record FutureType(List<Type> types) implements Type {

    public FutureType {
        types = List.copyOf(types);
    }

    @Override
    public String mlir() {
        return types.stream()
                .map(Type::mlir)
                .collect(Collectors.joining(", ", "!stablehlo.future<", ">"));
    }

    @Override
    public String toString() {
        return mlir();
    }
}
