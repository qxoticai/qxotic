package com.qxotic.jota;

import com.qxotic.jota.impl.NestedTuple;
import com.qxotic.jota.impl.ShapeFactory;
import java.util.Objects;

/**
 * An immutable tuple of dimension sizes, possibly nested: {@code Shape.of(2, Shape.of(3, 4), 5)}
 * has rank 3 and flat rank 4. A shape is unit-agnostic; views use physical storage shapes, which
 * differ from logical shapes for block-quantized data types (see {@link DataType}). Axis indexes
 * wrap around: a negative index counts from the end.
 */
public interface Shape extends NestedTuple<Shape> {

    /** The size of one top-level mode (wrap-around index). */
    long size(int _modeIndex);

    /** The total element count: the product of all flat dimensions. */
    long size();

    /** This shape with nesting removed (same flat dims, rank = flat rank). */
    Shape flattenModes();

    default boolean hasZeroElements() {
        return size() == 0;
    }

    default boolean hasOneElement() {
        return size() == 1;
    }

    /** A flat (non-nested) shape from dimension sizes. */
    static Shape flat(long... dims) {
        return ShapeFactory.flat(dims);
    }

    /**
     * A possibly nested shape from dimension sizes and nested shapes: {@code of(2, Shape.of(3, 4),
     * 5)}. The empty call returns {@link #scalar()}.
     */
    static Shape of(Object... elements) {
        return ShapeFactory.of(elements);
    }

    /**
     * A shape whose nesting is given in bracket notation: {@code pattern("(_,(b,c))", 2, 3, 4)} is
     * {@code of(2, of(3, 4))}.
     */
    static Shape pattern(String pattern, long... dims) {
        return ShapeFactory.pattern(pattern, dims);
    }

    /**
     * A shape with {@code template}'s nesting structure and the given flat dims, in flatten order.
     */
    static Shape template(NestedTuple<?> template, long... dims) {
        return ShapeFactory.template(template, dims);
    }

    /** The rank-0 shape: one element, no axes. */
    static Shape scalar() {
        return ShapeFactory.scalar();
    }

    /**
     * Alias for {@link #of(Object...)} designed for static import, enabling a DSL-style syntax for
     * constructing nested shapes:
     *
     * <pre>{@code
     * import static com.qxotic.jota.Shape.shape;
     *
     * Shape s = shape(2, shape(3, 4), 5);
     * }</pre>
     */
    static Shape shape(Object... elements) {
        return of(elements);
    }

    /**
     * Resolves a shape from dimensions that may contain a single {@code -1} placeholder.
     *
     * <p>Rules:
     *
     * <ul>
     *   <li>At most one dimension may be {@code -1}
     *   <li>All other dimensions must be {@code >= 1} (zero is not allowed)
     *   <li>Without {@code -1}, the target size must exactly match {@code totalSize}
     *   <li>With {@code -1}, inferred dimension is {@code totalSize / knownProduct} and must divide
     *       exactly
     * </ul>
     */
    static Shape resolveShape(long totalSize, long... dims) {
        Objects.requireNonNull(dims, "dims");
        if (totalSize < 0) {
            throw new IllegalArgumentException("totalSize must be >= 0");
        }
        if (dims.length == 0) {
            throw new IllegalArgumentException("resolveShape requires at least one dimension");
        }

        int inferIndex = -1;
        long knownProduct = 1L;
        for (int i = 0; i < dims.length; i++) {
            long dim = dims[i];
            if (dim == -1L) {
                if (inferIndex >= 0) {
                    throw new IllegalArgumentException(
                            "resolveShape allows at most one -1 dimension");
                }
                inferIndex = i;
                continue;
            }
            if (dim <= 0L) {
                throw new IllegalArgumentException(
                        "resolveShape dimensions must be >= 1 (or -1), got " + dim);
            }
            knownProduct = Math.multiplyExact(knownProduct, dim);
        }

        long[] resolved = dims.clone();
        if (inferIndex >= 0) {
            if (totalSize == 0L) {
                throw new IllegalArgumentException("cannot infer -1 for totalSize=0");
            }
            if (knownProduct == 0L || totalSize % knownProduct != 0L) {
                throw new IllegalArgumentException(
                        "cannot infer -1: totalSize "
                                + totalSize
                                + " is not divisible by known product "
                                + knownProduct);
            }
            long inferred = totalSize / knownProduct;
            if (inferred <= 0L) {
                throw new IllegalArgumentException("inferred dimension must be >= 1");
            }
            resolved[inferIndex] = inferred;
        } else if (knownProduct != totalSize) {
            throw new IllegalArgumentException(
                    "resolveShape size mismatch: target size="
                            + knownProduct
                            + " does not match totalSize="
                            + totalSize);
        }

        return Shape.flat(resolved);
    }
}
