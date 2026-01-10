package com.qxotic.jota.impl;

import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;
import com.qxotic.jota.Stride;

import java.util.Objects;

final class LayoutImpl implements Layout {

    private static final Layout SCALAR = LayoutImpl.of(Shape.scalar(), Stride.scalar());

    private final Shape shape;
    private final Stride stride;

    private LayoutImpl(Shape shape, Stride stride) {
        Objects.requireNonNull(shape);
        Objects.requireNonNull(stride);

        if (shape.flatRank() != stride.flatRank()) {
            throw new IllegalArgumentException(
                "Shape and stride must have the same flatRank: " +
                "shape.flatRank()=" + shape.flatRank() +
                " stride.flatRank()=" + stride.flatRank() +
                " (shape=" + shape + " stride=" + stride + ")");
        }

        // Note: We allow different nesting structures as long as flatRank matches,
        // following CuTe's design. Use isCongruent() to check structural equivalence.

        this.shape = shape;
        this.stride = stride;
    }

    public static Layout scalar() {
        return SCALAR;
    }

    @Override
    public Shape shape() {
        return this.shape;
    }

    @Override
    public Stride stride() {
        return this.stride;
    }

    @Override
    public Layout modeAt(int modeIndex) {
        Shape modeShape = shape.modeAt(modeIndex);
        Stride modeStride = stride.modeAt(modeIndex);
        return new LayoutImpl(modeShape, modeStride);
    }

    @Override
    public Layout flatten() {
        if (shape.isFlat() && stride.isFlat()) {
            return this;
        }
        return new LayoutImpl(shape.flatten(), stride.flatten());
    }

    @Override
    public boolean isCongruentWith(Layout other) {
        Objects.requireNonNull(other);
        return this.shape.isCongruentWith(other.shape())
            && this.stride.isCongruentWith(other.stride());
    }

    @Override
    public String toString() {
        return shape.toString() + ":" + stride.toString();
    }

    @Override
    public boolean equals(Object other) {
        return (other instanceof Layout that)
                && Objects.equals(this.shape, that.shape())
                && Objects.equals(this.stride, that.stride());
    }

    @Override
    public int hashCode() {
        return Objects.hash(shape, stride);
    }

    public static Layout of(Shape shape, Stride stride) {
        return new LayoutImpl(shape, stride);
    }
}
