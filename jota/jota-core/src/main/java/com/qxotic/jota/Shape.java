package com.qxotic.jota;

import com.qxotic.jota.impl.NestedTuple;
import com.qxotic.jota.impl.ShapeFactory;
import java.util.Objects;

public interface Shape extends NestedTuple<Shape> {

    long size(int _modeIndex);

    long size();

    Shape flattenModes();

    default boolean hasZeroElements() {
        return size() == 0;
    }

    default boolean hasOneElement() {
        return size() == 1;
    }

    static Shape flat(long... dims) {
        return ShapeFactory.flat(dims);
    }

    static Shape of(Object... elements) {
        return ShapeFactory.of(elements);
    }

    static Shape pattern(String pattern, long... dims) {
        return ShapeFactory.pattern(pattern, dims);
    }

    static Shape template(NestedTuple<?> template, long... dims) {
        return ShapeFactory.template(template, dims);
    }

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
