package com.llm4j.jota.memory;

import com.llm4j.jota.DataType;
import com.llm4j.jota.Shape;

import java.util.Arrays;
import java.util.stream.IntStream;

public interface MemoryView<B> {
    Shape shape();

    long[] byteStrides();

    DataType dataType();

    long byteOffset();

    Memory<B> memory();

    default boolean isBroadcasted() {
        return Arrays.stream(byteStrides()).anyMatch(byteStride -> byteStride == 0L);
    }

    default boolean isContiguous() {
        if (shape().hasZeroElements()) {
            return true; // Empty views are trivially contiguous
        }
        long expectedStride = dataType().byteSize();
        long[] strides = byteStrides();
        for (int i = shape().rank() - 1; i >= 0; i--) {
            if (strides[i] != expectedStride) {
                return false;
            }
            expectedStride *= shape().dimension(i);
        }
        return true;
    }

    MemoryView<B> reshape(Shape newShape);

    MemoryView<B> permute(int... permutationIndices);

    MemoryView<B> expand(Shape newShape);

    MemoryView<B> slice(int _axis, long fromInclusive, long toExclusive, long indexStride);

    default MemoryView<B> slice(int _axis, long fromInclusive, long toExclusive) {
        return slice(_axis, fromInclusive, toExclusive, 1);
    }

    default MemoryView<B> transpose(int _axis0, int _axis1) {
        Shape shape = shape();
        int axis0 = shape.wrapAround(_axis0);
        int axis1 = shape.wrapAround(_axis1);
        int[] permutation = IntStream.range(0, shape.rank()).toArray();
        permutation[axis0] = axis1;
        permutation[axis1] = axis0;
        return permute(permutation);
    }

    default MemoryView<B> broadcast(Shape targetShape) {
        // Step 1: Reshape by adding singleton dimensions if needed (without copying)
        Shape currentShape = shape();

        // Determine how many dimensions we need to add
        int numNewDims = targetShape.rank() - currentShape.rank();
        if (numNewDims < 0) {
            throw new IllegalArgumentException("Cannot broadcast shape " + currentShape
                    + " to shape " + targetShape + ": has fewer dimensions");
        }

        // Prepend singleton dimensions
        long[] newDims = new long[targetShape.rank()];
        for (int i = 0; i < numNewDims; i++) {
            newDims[i] = 1;
        }
        System.arraycopy(
                currentShape.toArray(), 0,
                newDims, numNewDims,
                currentShape.rank()
        );

        // Create the reshaped view
        MemoryView<B> reshaped = this.reshape(Shape.of(newDims));

        // Step 2: Expand singleton dimensions to target sizes
        return reshaped.expand(targetShape);
    }
}

