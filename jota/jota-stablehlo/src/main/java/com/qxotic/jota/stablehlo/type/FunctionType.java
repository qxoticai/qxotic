package com.qxotic.jota.stablehlo.type;

import java.util.List;
import java.util.stream.Collectors;

/** {@code (inputs) -> (outputs)}. */
public record FunctionType(List<Type> inputs, List<Type> results) implements Type {

    public FunctionType {
        inputs = List.copyOf(inputs);
        results = List.copyOf(results);
    }

    @Override
    public String mlir() {
        String in = inputs.stream().map(Type::mlir).collect(Collectors.joining(", ", "(", ")"));
        String out;
        if (results.size() == 1 && !(results.getFirst() instanceof FunctionType)) {
            out = results.getFirst().mlir();
        } else {
            out = results.stream().map(Type::mlir).collect(Collectors.joining(", ", "(", ")"));
        }
        return in + " -> " + out;
    }

    @Override
    public String toString() {
        return mlir();
    }
}
