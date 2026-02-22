package com.qxotic.jota;

import com.qxotic.jota.impl.LayoutFactory;

public interface Layout {
    Shape shape();

    Stride stride();

    Layout modeAt(int modeIndex);

    default Layout flatten() {
        if (shape().isFlat() && stride().isFlat()) {
            return this;
        }
        return of(shape().flatten(), stride().flatten());
    }

    boolean isCongruentWith(Layout other);

    /**
     * Returns true if the given mode (wrap-around index) is contiguous within its own mode range.
     * This checks contiguity only inside the mode and ignores any outer modes.
     *
     * <p>Example: shape = (A, (B, C), D), stride = [B*C*D*2, C*D, D, 2]. Mode 1 (B, C) is locally
     * contiguous because strides inside the mode follow row-major expectations when treating D as
     * the inner element size, even though axis D is strided.
     */
    default boolean isLocalContiguous(int _modeIndex) {
        int modeIndex = Util.wrapAround(_modeIndex, shape().rank());
        int startFlat = 0;
        for (int mode = 0; mode < modeIndex; mode++) {
            startFlat += shape().modeAt(mode).flatRank();
        }
        int modeFlatRank = shape().modeAt(modeIndex).flatRank();
        int endFlat = startFlat + modeFlatRank;
        if (modeFlatRank == 0) {
            return true;
        }
        long trailingScale = 1;
        for (int i = shape().flatRank() - 1; i >= endFlat; i--) {
            trailingScale *= shape().flatAt(i);
        }
        long expected = trailingScale;
        for (int i = endFlat - 1; i >= startFlat; i--) {
            if (stride().flatAt(i) != expected) {
                return false;
            }
            expected *= shape().flatAt(i);
        }
        return true;
    }

    /**
     * Returns true if the given mode (wrap-around index) and all inner flat axes are contiguous.
     * This is stricter than {@link #isLocalContiguous(int)} and requires the mode to be a suffix of
     * the flattened layout.
     *
     * <p>Example: shape = (A, (B, C), D). Mode 1 (B, C) can still be locally contiguous because its
     * internal strides match row-major when treated as a block, but suffix contiguity will be false
     * if axis D is strided (e.g., stride[D] != 1).
     */
    default boolean isSuffixContiguous(int _modeIndex) {
        int modeIndex = Util.wrapAround(_modeIndex, shape().rank());
        int startFlat = 0;
        for (int mode = 0; mode < modeIndex; mode++) {
            startFlat += shape().modeAt(mode).flatRank();
        }
        long expected = 1;
        for (int i = shape().flatRank() - 1; i >= startFlat; i--) {
            if (stride().flatAt(i) != expected) {
                return false;
            }
            expected *= shape().flatAt(i);
        }
        return true;
    }

    /**
     * Flat-axis version of {@link #isLocalContiguous(int)}. The flat axis is treated as a mode in
     * the flattened layout.
     */
    default boolean isLocalContiguousFlat(int _flatIndex) {
        return flatten().isLocalContiguous(_flatIndex);
    }

    /**
     * Flat-axis version of {@link #isSuffixContiguous(int)}. The flat axis is treated as a mode in
     * the flattened layout.
     */
    default boolean isSuffixContiguousFlat(int _flatIndex) {
        return flatten().isSuffixContiguous(_flatIndex);
    }

    static Layout of(Shape shape, Stride stride) {
        return LayoutFactory.of(shape, stride);
    }

    static Layout scalar() {
        return LayoutFactory.scalar();
    }

    static Layout rowMajor(Shape shape) {
        return of(shape, Stride.rowMajor(shape));
    }

    static Layout columnMajor(Shape shape) {
        return of(shape, Stride.columnMajor(shape));
    }

    static Layout rowMajor(long... dims) {
        Shape shape = Shape.flat(dims);
        return of(shape, Stride.rowMajor(shape));
    }

    static Layout columnMajor(long... dims) {
        Shape shape = Shape.flat(dims);
        return of(shape, Stride.columnMajor(shape));
    }

    static Layout nestedLikeShape(Shape shape, Stride stride) {
        return LayoutFactory.of(shape, Stride.template(shape, stride.toArray()));
    }

    static Layout nestedLikeStride(Stride stride, Shape shape) {
        return LayoutFactory.of(Shape.template(stride, shape.toArray()), stride);
    }

    /**
     * Returns true if this layout spans a contiguous memory range [0, n-1].
     *
     * <p>This is the CuTe-style contiguity check: a layout is "contiguous" if all elements fit
     * within a contiguous block of memory without gaps. Formally:
     *
     * <pre>sum((dim_i - 1) * stride_i) == totalElements - 1</pre>
     *
     * <p>This is more general than row-major contiguity. For example, (2, 2, 2):(4, 1, 2) spans [0,
     * 7] contiguously even though iteration order is not linear.
     *
     * <p>Empty layouts (size == 0) are considered contiguous by convention.
     */
    default boolean spansContiguousRange() {
        return isSpanContiguous();
    }

    /**
     * Returns true if this layout spans a gapless memory range.
     *
     * <p>This is a compactness check: the accessed offsets form a contiguous span, even if the
     * iteration order is not row-major.
     */
    default boolean isSpanContiguous() {
        if (shape().hasZeroElements()) {
            return true;
        }
        long totalElements = 1;
        long minOffset = 0;
        long maxOffset = 0;
        long[] strides = stride().toArray();
        for (int i = 0; i < shape().flatRank(); i++) {
            long dim = shape().flatAt(i);
            if (dim <= 1) {
                continue;
            }
            long strideValue = strides[i];
            long span = (dim - 1) * strideValue;
            if (strideValue >= 0) {
                maxOffset += span;
            } else {
                minOffset += span;
            }
            totalElements *= dim;
        }
        return (maxOffset - minOffset) == totalElements - 1;
    }

    /**
     * Returns true if this layout is contiguous in row-major order.
     *
     * <p>This is a stricter check than {@link #spansContiguousRange()}. A layout is row-major
     * contiguous if:
     *
     * <ol>
     *   <li>The rightmost (innermost) stride equals 1
     *   <li>Each stride equals the product of all inner dimensions times their strides
     * </ol>
     *
     * <p>Empty layouts (size == 0) are considered contiguous by convention.
     *
     * @deprecated Use {@link #spansContiguousRange()} for CuTe semantics, or check {@link
     *     #isSuffixContiguous(int)} for specific modes.
     */
    @Deprecated
    default boolean isContiguous() {
        return spansContiguousRange();
    }

    /** Returns true if this layout is contiguous in row-major order. */
    default boolean isRowMajorContiguous() {
        if (shape().hasZeroElements()) {
            return true;
        }
        long expected = 1;
        for (int i = shape().flatRank() - 1; i >= 0; i--) {
            long dim = shape().flatAt(i);
            if (dim <= 1) {
                continue;
            }
            if (stride().flatAt(i) != expected) {
                return false;
            }
            expected *= dim;
        }
        return true;
    }

    /** Returns true if this layout has no overlapping indices (no broadcast/aliasing). */
    default boolean isNonOverlapping() {
        if (shape().hasZeroElements()) {
            return true;
        }
        int rank = shape().flatRank();
        long[] dims = shape().toArray();
        long[] strides = stride().toArray();

        int count = 0;
        for (int i = 0; i < rank; i++) {
            if (dims[i] > 1) {
                if (strides[i] == 0) {
                    return false;
                }
                count++;
            }
        }
        if (count <= 1) {
            return true;
        }

        Integer[] order = new Integer[rank];
        for (int i = 0; i < rank; i++) {
            order[i] = i;
        }
        java.util.Arrays.sort(
                order, (a, b) -> Long.compare(Math.abs(strides[a]), Math.abs(strides[b])));

        long required = 1;
        for (int idx : order) {
            long dim = dims[idx];
            if (dim <= 1) {
                continue;
            }
            long strideAbs = Math.abs(strides[idx]);
            if (strideAbs < required) {
                return false;
            }
            required = strideAbs * dim;
        }
        return true;
    }
}
