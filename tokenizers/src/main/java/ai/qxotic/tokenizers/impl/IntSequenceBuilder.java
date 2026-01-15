package ai.qxotic.tokenizers.impl;

import ai.qxotic.tokenizers.IntSequence;

import java.util.Arrays;

/**
 * A builder for creating IntSequences.
 * The build is an IntSequence itself, a view over unmodifiable data.
 */
final class IntSequenceBuilder extends AbstractIntSequence implements IntSequence.Builder {
    private int[] data;
    private int size;

    private static final int DEFAULT_CAPACITY = 8;
    private static final int[] DEFAULTCAPACITY_EMPTY_ELEMENTDATA = {};

    private IntSequenceBuilder(int[] data, int size) {
        this.data = data;
        this.size = size;
    }

    IntSequenceBuilder(int capacity) {
        if (capacity < 0) {
            throw new IllegalArgumentException("negative capacity");
        }
        this.data = new int[Math.max(1, capacity)];
        this.size = 0;
    }

    IntSequenceBuilder() {
        this.data = DEFAULTCAPACITY_EMPTY_ELEMENTDATA;
        this.size = 0;
    }

    @Override
    public IntSequenceBuilder add(int value) {
        if (this.size == data.length) {
            grow(size + 1);
        }
        this.data[this.size++] = value;
        return this;
    }

    @Override
    public void ensureCapacity(int minCapacity) {
        if (minCapacity < 0) {
            throw new IllegalArgumentException("negative capacity");
        }
        if (minCapacity > data.length
                && !(data == DEFAULTCAPACITY_EMPTY_ELEMENTDATA
                && minCapacity <= DEFAULT_CAPACITY)) {
            grow(minCapacity);
        }
    }

    private static final int SOFT_MAX_ARRAY_LENGTH = Integer.MAX_VALUE - 8;

    private static int newLength(int oldLength, int minGrowth, int prefGrowth) {
        // preconditions not checked because of inlining
        // assert oldLength >= 0
        // assert minGrowth > 0
        int prefLength = oldLength + Math.max(minGrowth, prefGrowth); // might overflow
        if (0 < prefLength && prefLength <= SOFT_MAX_ARRAY_LENGTH) {
            return prefLength;
        } else {
            // put code cold in a separate method
            return hugeLength(oldLength, minGrowth);
        }
    }

    private static int hugeLength(int oldLength, int minGrowth) {
        int minLength = oldLength + minGrowth;
        if (minLength < 0) { // overflow
            throw new OutOfMemoryError(
                    "Required array length " + oldLength + " + " + minGrowth + " is too large");
        } else if (minLength <= SOFT_MAX_ARRAY_LENGTH) {
            return SOFT_MAX_ARRAY_LENGTH;
        } else {
            return minLength;
        }
    }

    /**
     * Increases the capacity to ensure that it can hold at least the
     * number of elements specified by the minimum capacity argument.
     *
     * @param minCapacity the desired minimum capacity
     * @throws OutOfMemoryError if minCapacity is less than zero
     */
    private int[] grow(int minCapacity) {
        int oldCapacity = this.data.length;
        if (oldCapacity > 0 || this.data != DEFAULTCAPACITY_EMPTY_ELEMENTDATA) {
            int newCapacity = newLength(oldCapacity,
                    minCapacity - oldCapacity, /* minimum growth */
                    oldCapacity >> 1           /* preferred growth */);
            return this.data = Arrays.copyOf(this.data, newCapacity);
        } else {
            return this.data = new int[Math.max(DEFAULT_CAPACITY, minCapacity)];
        }
    }

    public IntSequence build() {
        return new ArrayIntSequence(Arrays.copyOf(this.data, length()));
    }

    @Override
    public int intAt(int index) {
        if (index < 0 || index >= this.size) {
            throw new IndexOutOfBoundsException("Index: " + index + ", Length: " + this.size);
        }
        return this.data[index];
    }

    @Override
    public int length() {
        return size;
    }

    @Override
    public IntSequence subSequence(int start, int end) {
        if (start < 0 || end < start || end > length()) {
            throw new IllegalArgumentException("Invalid subSequence range [" + start + ", " + end + ")");
        }
        if (start == end) {
            return ImplAccessor.empty();
        }
        return new ArrayIntSequence(this.data, start, end - start);
    }

    @Override
    public int[] toArray() {
        return Arrays.copyOf(this.data, this.size);
    }
}
