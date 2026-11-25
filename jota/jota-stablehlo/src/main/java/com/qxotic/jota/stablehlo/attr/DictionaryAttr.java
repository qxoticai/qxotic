package com.qxotic.jota.stablehlo.attr;

import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

/** {@code {name = value, ...}}; entries are kept sorted by name, as in MLIR. */
public record DictionaryAttr(List<NamedAttribute> entries) implements Attribute {

    public static final DictionaryAttr EMPTY = new DictionaryAttr(List.of());

    public DictionaryAttr {
        entries =
                entries.stream()
                        .sorted(java.util.Comparator.comparing(NamedAttribute::name))
                        .toList();
    }

    public Optional<Attribute> get(String name) {
        for (NamedAttribute e : entries) {
            if (e.name().equals(name)) {
                return Optional.of(e.value());
            }
        }
        return Optional.empty();
    }

    public boolean isEmpty() {
        return entries.isEmpty();
    }

    @Override
    public String toString() {
        return entries.stream()
                .map(NamedAttribute::toString)
                .collect(Collectors.joining(", ", "{", "}"));
    }
}
