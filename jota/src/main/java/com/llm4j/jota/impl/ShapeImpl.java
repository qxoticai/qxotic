package com.llm4j.jota.impl;

import com.llm4j.jota.Shape;

import java.util.Arrays;
import java.util.Objects;
import java.util.stream.LongStream;

final class ShapeImpl implements Shape {

    static final Shape SCALAR = new ShapeImpl(new long[0]);

    // @CompilationFinal(dimensions = 1)
    private final long[] dimensions;
    private final long totalNumberOfElements;

    // Package-private constructor to enforce encapsulation.
    ShapeImpl(long[] dimensions) {
        this.dimensions = validateDimensions(dimensions); // returns a copy
        this.totalNumberOfElements = LongStream.of(dimensions).reduce(1L, Math::multiplyExact);
    }

    @Override
    public int rank() {
        return dimensions.length;
    }

    @Override
    public long dimension(int _axis) {
        return dimensions[wrapAround(_axis)];
    }

    @Override
    public long totalNumberOfElements() {
        return totalNumberOfElements;
    }

    // Validates the dimensions and returns a defensive copy
    private static long[] validateDimensions(long[] dimensions) {
        Objects.requireNonNull(dimensions);
        for (int i = 0; i < dimensions.length; i++) {
            if (dimensions[i] < 0) {
                throw new IllegalArgumentException("Negative dimension at index : " + i);
            }
        }
        return dimensions.clone();
    }

    @Override
    public long[] toArray() {
        return dimensions.clone();
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        // Fast-path.
        if (obj instanceof ShapeImpl thatShapeImpl) {
            return Arrays.equals(this.dimensions, thatShapeImpl.dimensions);
        }
        // Fallback to generic Shape equals.
        return (obj instanceof Shape thatShape) && Shape.sameAs(this, thatShape);
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(dimensions);
    }

    @Override
    public String toString() {
        return Arrays.toString(dimensions);
    }
}
