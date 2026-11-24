package com.qxotic.jota.stablehlo.attr;

import java.util.List;
import java.util.stream.Collectors;

/** {@code [a, b, ...]}. */
public record ArrayAttr(List<Attribute> elements) implements Attribute {

    public ArrayAttr {
        elements = List.copyOf(elements);
    }

    @Override
    public String toString() {
        return elements.stream().map(Object::toString).collect(Collectors.joining(", ", "[", "]"));
    }
}
