package com.qxotic.toknroll.impl;

import com.qxotic.toknroll.IntSequence;
import java.util.Arrays;

/** A builder for creating {@link IntSequence} instances. */
final class IntSequenceBuilder implements IntSequence.Builder {
    private static final int DEFAULT_CAPACITY = 8;
    private static final int MAX_ARRAY_SIZE = Integer.MAX_VALUE - 8;
    private static final int[] EMPTY = {};

    private int[] data;
    private int size;

    IntSequenceBuilder(int capacity) {
        if (capacity < 0) {
            throw new IllegalArgumentException("negative capacity");
        }
        this.data = capacity == 0 ? EMPTY : new int[capacity];
        this.size = 0;
    }

    IntSequenceBuilder() {
        this.data = EMPTY;
        this.size = 0;
    }

    @Override
    public IntSequenceBuilder add(int value) {
        ensureCapacity(size + 1);
        this.data[this.size++] = value;
        return this;
    }

    @Override
    public void ensureCapacity(int minCapacity) {
        if (minCapacity < 0) {
            throw new IllegalArgumentException("negative capacity");
        }
        if (minCapacity > data.length) {
            grow(minCapacity);
        }
    }

    private void grow(int minCapacity) {
        int oldCapacity = data.length;
        int newCapacity =
                oldCapacity == 0
                        ? Math.max(DEFAULT_CAPACITY, minCapacity)
                        : oldCapacity + (oldCapacity >> 1);
        if (newCapacity < minCapacity) {
            newCapacity = minCapacity;
        }
        if (newCapacity > MAX_ARRAY_SIZE) {
            newCapacity = hugeCapacity(minCapacity);
        }
        // an empty builder has nothing to carry over: a fresh array, not a copy of zeros
        data = size == 0 ? new int[newCapacity] : Arrays.copyOf(data, newCapacity);
    }

    private static int hugeCapacity(int minCapacity) {
        if (minCapacity < 0) {
            throw new OutOfMemoryError("Required array size too large");
        }
        return minCapacity > MAX_ARRAY_SIZE ? Integer.MAX_VALUE : MAX_ARRAY_SIZE;
    }

    public IntSequence build() {
        // The sequence keeps the array when most of it is in use: nothing here ever writes below
        // the size again, so the sequence stays what it was. A sparse array is copied down.
        if (size >= (data.length >>> 1)) {
            return new ArrayIntSequence(data, 0, size);
        }
        return new ArrayIntSequence(Arrays.copyOf(this.data, size));
    }

    @Override
    public int size() {
        return size;
    }

    /** The backing array, for a writer that has ensured the capacity and will set the size. */
    int[] data() {
        return data;
    }

    /** Takes {@code data} over as the backing array, its first {@code size} ids in use. */
    void adopt(int[] data, int size) {
        this.data = data;
        this.size = size;
    }

    @Override
    public IntSequence snapshot() {
        return new BuilderSequenceView(this, 0, size, false);
    }

    @Override
    public IntSequence asSequenceView() {
        return new BuilderSequenceView(this, 0, 0, true);
    }

    /** The element at {@code index}, for a caller in this package that must not allocate a view. */
    int get(int index) {
        if (index < 0 || index >= size) {
            throw new IndexOutOfBoundsException("index " + index + " of " + size);
        }
        return data[index];
    }

    private static final class BuilderSequenceView extends AbstractIntSequence {
        private final IntSequenceBuilder builder;
        private final int offset;
        private final int fixedLength;
        private final boolean live;

        private BuilderSequenceView(
                IntSequenceBuilder builder, int offset, int fixedLength, boolean live) {
            this.builder = builder;
            this.offset = offset;
            this.fixedLength = fixedLength;
            this.live = live;
        }

        @Override
        public int intAt(int index) {
            int length = length();
            if (index < 0 || index >= length) {
                throw new IndexOutOfBoundsException("Index: " + index + ", Length: " + length);
            }
            return builder.data[offset + index];
        }

        @Override
        public int length() {
            return live ? builder.size - offset : fixedLength;
        }

        @Override
        public IntSequence subSequence(int startInclusive, int endExclusive) {
            int length = length();
            if (startInclusive < 0 || endExclusive < startInclusive || endExclusive > length) {
                throw new IndexOutOfBoundsException(
                        "Invalid subSequence range [" + startInclusive + ", " + endExclusive + ")");
            }
            if (startInclusive == endExclusive) {
                return ImplAccessor.empty();
            }
            return new BuilderSequenceView(
                    builder, offset + startInclusive, endExclusive - startInclusive, false);
        }

        @Override
        public int[] toArray() {
            int length = length();
            return Arrays.copyOfRange(builder.data, offset, offset + length);
        }
    }
}
