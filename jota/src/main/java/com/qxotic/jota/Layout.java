package com.qxotic.jota;

import com.qxotic.jota.impl.LayoutFactory;

public interface Layout {
    Shape shape();

    Stride stride();

    Layout modeAt(int modeIndex);

    Layout flatten();

    boolean isCongruentWith(Layout other);

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
