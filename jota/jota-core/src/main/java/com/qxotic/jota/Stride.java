package com.qxotic.jota;

import com.qxotic.jota.impl.NestedTuple;
import com.qxotic.jota.impl.StrideFactory;

/**
 * Step sizes per axis, mirroring a {@link Shape}'s nesting: coordinate {@code c} maps to layout
 * offset {@code Σ c[i] × stride[i]}. Strides use the layout's units (elements for dense data, whole
 * blocks for block-quantized data). A zero stride broadcasts its axis. Immutable.
 */
public interface Stride extends NestedTuple<Stride> {

    /** The rank-0 stride. */
    static Stride scalar() {
        return StrideFactory.scalar();
    }

    /** A flat (non-nested) stride. */
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

    /** A possibly nested stride from step sizes and nested strides; see {@link Shape#of}. */
    static Stride of(Object... elements) {
        return StrideFactory.of(elements);
    }

    /**
     * A stride with {@code template}'s nesting structure and the given flat values, in flatten
     * order.
     */
    static Stride template(NestedTuple<?> template, long... strides) {
        return StrideFactory.template(template, strides);
    }

    /** A stride whose nesting is given in bracket notation; see {@link Shape#pattern}. */
    static Stride pattern(String pattern, long... strides) {
        return StrideFactory.pattern(pattern, strides);
    }

    /** Compact row-major (C-order) strides for {@code shape}, preserving its nesting. */
    static Stride rowMajor(Shape shape) {
        return computeStrides(shape, false);
    }

    /** Compact column-major (Fortran-order) strides for {@code shape}, preserving its nesting. */
    static Stride columnMajor(Shape shape) {
        return computeStrides(shape, true);
    }

    /** Multiplies every step by {@code factor}, preserving nested structure. */
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

    /**
     * Alias for {@link #of(Object...)} designed for static import, enabling a DSL-style syntax for
     * constructing nested strides:
     *
     * <pre>{@code
     * import static com.qxotic.jota.Stride.stride;
     *
     * Stride s = stride(1, stride(4, 5));
     * }</pre>
     */
    static Stride stride(Object... elements) {
        return of(elements);
    }

    /**
     * Returns true if all stride values are zero. Used for validating scalar constant tensors that
     * can be broadcasted without allocation.
     */
    default boolean isAllZeros() {
        for (int i = 0; i < flatRank(); i++) {
            if (flatAt(i) != 0) {
                return false;
            }
        }
        return true;
    }
}
