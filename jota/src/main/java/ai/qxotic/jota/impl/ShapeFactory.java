package ai.qxotic.jota.impl;

import ai.qxotic.jota.Shape;

public final class ShapeFactory {

    public static Shape flat(long... dims) {
        return ShapeImpl.flat(dims);
    }

    public static Shape pattern(String pattern, long... dims) {
        try {
            int[] nest = PatternParser.parsePattern(pattern, dims.length, "dimension");
            if (dims.length == 0) {
                return ShapeImpl.scalar();
            }
            return ShapeImpl.of(dims, nest);
        } catch (ArrayIndexOutOfBoundsException e) {
            throw new IllegalArgumentException("Pattern has more elements than provided dimensions", e);
        }
    }

    public static Shape template(NestedTuple<?> template, long... dims) {
        if (template instanceof NestedTupleImpl<?> impl) {
            if (impl.flatRank() != dims.length) {
                throw new IllegalArgumentException(
                    "Template has " + impl.flatRank() + " dimensions but " + dims.length + " were provided");
            }
            if (dims.length == 0) {
                return ShapeImpl.scalar();
            }
            return ShapeImpl.of(dims, impl.nest);
        }
        throw new IllegalArgumentException("Unsupported NestedTuple implementation");
    }

    public static Shape of(Object... elements) {
        if (elements.length == 0) {
            return ShapeImpl.scalar();
        }

        // Composition: of(Number/Shape... elements)
        return ShapeImpl.nested(elements);
    }

    public static Shape scalar() {
        return ShapeImpl.scalar();
    }
}
