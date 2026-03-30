package com.qxotic.tokenizers.impl;

import com.qxotic.tokenizers.IntSequence;
import java.util.Arrays;
import java.util.Objects;

/** An implementation of IntSequence backed by an array. */
final class ArrayIntSequence extends AbstractIntSequence {

    private final int[] array;
    private final int offset;
    private final int length;

    /**
     * Creates a new ArrayIntSequence backed by the given array.
     *
     * @param array the source array
     */
    ArrayIntSequence(int[] array) {
        this(array, 0, array.length);
    }

    /**
     * Creates a new ArrayIntSequence backed by a slice of the given array.
     *
     * @param array the source array
     * @param offset the starting offset in the array
     * @param length the length of the sequence
     */
    ArrayIntSequence(int[] array, int offset, int length) {
        Objects.requireNonNull(array, "array");
        if (offset < 0) {
            throw new IllegalArgumentException("Offset cannot be negative");
        }
        if (length < 0) {
            throw new IllegalArgumentException("Length cannot be negative");
        }
        if (offset > array.length - length) {
            throw new IllegalArgumentException("Offset + length exceeds array bounds");
        }

        this.array = array;
        this.offset = offset;
        this.length = length;
    }

    @Override
    public int intAt(int index) {
        if (index < 0 || index >= length) {
            throw new IndexOutOfBoundsException("Index: " + index + ", Length: " + length);
        }
        return array[offset + index];
    }

    @Override
    public int length() {
        return length;
    }

    @Override
    public IntSequence subSequence(int startInclusive, int endExclusive) {
        if (startInclusive < 0) {
            throw new IndexOutOfBoundsException("Start index cannot be negative");
        }
        if (endExclusive > length) {
            throw new IndexOutOfBoundsException("End index exceeds length");
        }
        if (startInclusive > endExclusive) {
            throw new IndexOutOfBoundsException("Start index greater than end index");
        }
        if (startInclusive == endExclusive) {
            return ImplAccessor.empty();
        }

        return new ArrayIntSequence(array, offset + startInclusive, endExclusive - startInclusive);
    }

    @Override
    public int[] toArray() {
        return Arrays.copyOfRange(array, offset, offset + length);
    }
}
