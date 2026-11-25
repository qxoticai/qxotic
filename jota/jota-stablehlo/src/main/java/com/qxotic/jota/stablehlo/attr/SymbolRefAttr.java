package com.qxotic.jota.stablehlo.attr;

import java.util.List;
import java.util.stream.Collectors;

/** {@code @root} or nested {@code @root::@leaf}. */
public record SymbolRefAttr(List<String> path) implements Attribute {

    public SymbolRefAttr {
        path = List.copyOf(path);
        if (path.isEmpty()) {
            throw new IllegalArgumentException("empty symbol reference");
        }
    }

    public static SymbolRefAttr of(String name) {
        return new SymbolRefAttr(List.of(name));
    }

    public String root() {
        return path.getFirst();
    }

    public String leaf() {
        return path.getLast();
    }

    static String symbol(String name) {
        return "@" + (AttrSyntax.isBareIdentifier(name) ? name : AttrSyntax.quote(name));
    }

    @Override
    public String toString() {
        return path.stream().map(SymbolRefAttr::symbol).collect(Collectors.joining("::"));
    }
}
