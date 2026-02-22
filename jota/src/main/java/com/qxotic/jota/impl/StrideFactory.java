package com.qxotic.jota.impl;

import com.qxotic.jota.Stride;

public final class StrideFactory {

    public static Stride flat(long... strides) {
        return StrideImpl.flat(strides);
    }

    public static Stride template(NestedTuple<?> template, long... strides) {
        if (template.flatRank() != strides.length) {
            throw new IllegalArgumentException(
                    "Template has "
                            + template.flatRank()
                            + " dimensions but "
                            + strides.length
                            + " were provided");
        }
        if (strides.length == 0) {
            return StrideImpl.scalar();
        }
        if (template instanceof NestedTupleImpl<?> impl) {
            return StrideImpl.of(strides, impl.nest);
        } else {
            throw new IllegalArgumentException("Unsupported NestedTuple implementation");
        }
    }

    public static Stride of(Object... elements) {
        if (elements.length == 0) {
            return StrideImpl.scalar();
        }

        // Composition: of(Number/Stride... elements)
        return StrideImpl.nested(elements);
    }

    public static Stride scalar() {
        return StrideImpl.scalar();
    }

    public static Stride zeros(int rank) {
        if (rank == 0) {
            return StrideImpl.scalar();
        }
        return StrideImpl.flat(new long[rank]); // All zeros
    }

    public static Stride zeros(NestedTuple<?> template) {
        int flatRank = template.flatRank();
        if (flatRank == 0) {
            return StrideImpl.scalar();
        }
        return template(template, new long[flatRank]); // All zeros with template structure
    }

    public static Stride pattern(String pattern, long... strides) {
        try {
            int[] nest = PatternParser.parsePattern(pattern, strides.length, "stride");
            if (strides.length == 0) {
                return StrideImpl.scalar();
            }
            return StrideImpl.of(strides, nest);
        } catch (ArrayIndexOutOfBoundsException e) {
            throw new IllegalArgumentException(
                    "Pattern has more elements than provided strides", e);
        }
    }
}
