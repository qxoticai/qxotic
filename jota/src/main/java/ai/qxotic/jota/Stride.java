package ai.qxotic.jota;

import ai.qxotic.jota.impl.NestedTuple;
import ai.qxotic.jota.impl.StrideFactory;

public interface Stride extends NestedTuple<Stride> {

    static Stride scalar() {
        return StrideFactory.scalar();
    }

    static Stride flat(long... strides) {
        return StrideFactory.flat(strides);
    }

    /** Creates a stride with all zeros (for broadcast/scalar tensors). */
    static Stride zeros(int rank) {
        return StrideFactory.zeros(rank);
    }

    /** Creates a stride with all zeros matching the template's structure. */
    static Stride zeros(NestedTuple<?> template) {
        return StrideFactory.zeros(template);
    }

    static Stride of(Object... elements) {
        return StrideFactory.of(elements);
    }

    static Stride template(NestedTuple<?> template, long... strides) {
        return StrideFactory.template(template, strides);
    }

    static Stride pattern(String pattern, long... strides) {
        return StrideFactory.pattern(pattern, strides);
    }

    static Stride rowMajor(Shape shape) {
        return computeStrides(shape, false);
    }

    static Stride columnMajor(Shape shape) {
        return computeStrides(shape, true);
    }

    /** Scale stride by factor (multiplies each element). Preserves nested structure. */
    Stride scale(long factor);

    private static Stride computeStrides(Shape shape, boolean columnMajor) {
        if (shape.isScalar()) {
            return Stride.of();
        }

        long[] strides = new long[shape.flatRank()];
        long accumulator = 1;

        if (columnMajor) {
            // Left to right
            for (int i = 0; i < shape.flatRank(); i++) {
                strides[i] = accumulator;
                accumulator *= shape.flatAt(i);
            }
        } else {
            // Right to left (row-major)
            for (int i = shape.flatRank() - 1; i >= 0; i--) {
                strides[i] = accumulator;
                accumulator *= shape.flatAt(i);
            }
        }

        if (shape.isFlat()) {
            return StrideFactory.flat(strides);
        }

        // Preserve nesting structure
        return StrideFactory.template(shape, strides);
    }

    static Stride stride(Object... elements) {
        return of(elements);
    }
}
