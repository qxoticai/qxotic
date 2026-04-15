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
        data = Arrays.copyOf(data, newCapacity);
    }

    private static int hugeCapacity(int minCapacity) {
        if (minCapacity < 0) {
            throw new OutOfMemoryError("Required array size too large");
        }
        return minCapacity > MAX_ARRAY_SIZE ? Integer.MAX_VALUE : MAX_ARRAY_SIZE;
    }

    public IntSequence build() {
        return new ArrayIntSequence(Arrays.copyOf(this.data, size));
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public IntSequence snapshot() {
        return new BuilderSequenceView(this, 0, size, false);
    }

    @Override
    public IntSequence asSequenceView() {
        return new BuilderSequenceView(this, 0, 0, true);
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
