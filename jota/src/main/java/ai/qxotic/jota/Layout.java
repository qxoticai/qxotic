package ai.qxotic.jota;

import ai.qxotic.jota.impl.LayoutFactory;

import ai.qxotic.jota.Util;

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
     * <p>Example: shape = (A, (B, C), D), stride = [B*C*D*2, C*D, D, 2].
     * Mode 1 (B, C) is locally contiguous because strides inside the mode follow row-major
     * expectations when treating D as the inner element size, even though axis D is strided.
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
     * This is stricter than {@link #isLocalContiguous(int)} and requires the mode to be a suffix
     * of the flattened layout.
     *
     * <p>Example: shape = (A, (B, C), D). Mode 1 (B, C) can still be locally contiguous because
     * its internal strides match row-major when treated as a block, but suffix contiguity will be
     * false if axis D is strided (e.g., stride[D] != 1).
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
     * Flat-axis version of {@link #isLocalContiguous(int)}. The flat axis is treated as a mode
     * in the flattened layout.
     */
    default boolean isLocalContiguousFlat(int _flatIndex) {
        return flatten().isLocalContiguous(_flatIndex);
    }

    /**
     * Flat-axis version of {@link #isSuffixContiguous(int)}. The flat axis is treated as a mode
     * in the flattened layout.
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
}
