package com.qxotic.toknroll;

import com.qxotic.toknroll.impl.ImplAccessor;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.PrimitiveIterator;
import java.util.Spliterator;
import java.util.Spliterators;
import java.util.StringJoiner;
import java.util.function.IntConsumer;
import java.util.stream.IntStream;
import java.util.stream.StreamSupport;

/**
 * A read-only sequence of {@code int} values.
 *
 * <p>Sequences are ordered lexicographically (element-by-element; a prefix sequence compares less
 * than any longer sequence that starts with it), consistent with {@link Arrays#compare(int[],
 * int[])}.
 */
public interface IntSequence extends Iterable<Integer>, Comparable<IntSequence> {

    /** Returns the shared empty sequence. */
    static IntSequence empty() {
        return ImplAccessor.empty();
    }

    /**
     * @param index element index
     * @return the element at {@code index}
     * @throws IndexOutOfBoundsException if the index is out of range
     */
    int intAt(int index);

    /** Returns the number of elements in this sequence. */
    int length();

    /**
     * Returns the half-open range {@code [startInclusive, endExclusive)} of this sequence.
     *
     * @throws IndexOutOfBoundsException if the range is out of bounds
     */
    IntSequence subSequence(int startInclusive, int endExclusive);

    /** Returns a new {@code int[]} copy of this sequence. */
    default int[] toArray() {
        int length = length();
        int[] array = new int[length];
        copyTo(0, array, 0, length);
        return array;
    }

    /** Returns a new {@code List} with all elements boxed as {@code Integer}. */
    default List<Integer> toList() {
        int length = length();
        List<Integer> list = new ArrayList<>(length);
        for (int i = 0; i < length; i++) {
            list.add(intAt(i));
        }
        return list;
    }

    /**
     * Returns a primitive iterator over this sequence.
     *
     * <p>The iterator captures the sequence length at creation time; elements appended afterwards
     * are not visited.
     */
    @Override
    default PrimitiveIterator.OfInt iterator() {
        final int iterationLength = length();
        return new PrimitiveIterator.OfInt() {
            private int index = 0;

            @Override
            public boolean hasNext() {
                return index < iterationLength;
            }

            @Override
            public int nextInt() {
                if (!hasNext()) {
                    throw new NoSuchElementException();
                }
                return intAt(index++);
            }
        };
    }

    /** Returns a sequential {@link IntStream} over this sequence. */
    default IntStream stream() {
        return StreamSupport.intStream(
                Spliterators.spliterator(
                        iterator(),
                        length(),
                        Spliterator.ORDERED | Spliterator.SIZED | Spliterator.SUBSIZED),
                false);
    }

    /** Returns a sequence copying {@code values}. */
    static IntSequence of(int... values) {
        if (values.length == 0) {
            return empty();
        }
        return copyOf(values);
    }

    /**
     * Wraps {@code integerList} without copying.
     *
     * <p>The returned sequence is unmodifiable through this API, but reflects subsequent mutations
     * to the wrapped list.
     */
    static IntSequence wrap(List<Integer> integerList) {
        return ImplAccessor.wrap(integerList);
    }

    /**
     * Wraps {@code array} without copying.
     *
     * <p>The returned sequence is unmodifiable through this API, but reflects subsequent mutations
     * to the wrapped array.
     */
    static IntSequence wrap(int[] array) {
        if (array.length == 0) {
            return empty(); // cannot grow
        }
        return ImplAccessor.wrap(array);
    }

    /**
     * Creates an IntSequence by copying the provided array.
     *
     * <p>Unlike {@link #wrap(int[])}, subsequent mutations to the original array are not reflected
     * in the returned sequence.
     */
    static IntSequence copyOf(int[] array) {
        Objects.requireNonNull(array, "array");
        if (array.length == 0) {
            return empty();
        }
        return ImplAccessor.wrap(Arrays.copyOf(array, array.length));
    }

    /**
     * Creates an IntSequence by copying the provided list values.
     *
     * <p>Unlike {@link #wrap(List)}, subsequent mutations to the original list are not reflected in
     * the returned sequence.
     */
    static IntSequence copyOf(List<Integer> integerList) {
        Objects.requireNonNull(integerList, "integerList");
        if (integerList.isEmpty()) {
            return empty();
        }
        int[] array = new int[integerList.size()];
        int i = 0;
        for (Integer value : integerList) {
            array[i++] = value;
        }
        return ImplAccessor.wrap(array);
    }

    /** Concatenates all provided sequences in order. */
    static IntSequence concatAll(IntSequence... sequences) {
        Objects.requireNonNull(sequences, "sequences");
        if (sequences.length == 0) {
            return IntSequence.empty();
        }
        int total = 0;
        for (IntSequence sequence : sequences) {
            total = Math.addExact(total, Objects.requireNonNull(sequence, "sequence").length());
        }
        if (total == 0) {
            return IntSequence.empty();
        }
        int[] merged = new int[total];
        int offset = 0;
        for (IntSequence sequence : sequences) {
            int length = sequence.length();
            sequence.copyTo(merged, offset, length);
            offset += length;
        }
        return IntSequence.wrap(merged);
    }

    /** Compares two sequences for content equality. */
    static boolean contentEquals(IntSequence first, IntSequence second) {
        if (Objects.requireNonNull(first) == Objects.requireNonNull(second)) {
            return true;
        }
        int length = first.length();
        if (length != second.length()) {
            return false;
        }
        for (int i = 0; i < length; i++) {
            if (first.intAt(i) != second.intAt(i)) {
                return false;
            }
        }
        return true;
    }

    /** Compares two sequences lexicographically. */
    static int compare(IntSequence first, IntSequence second) {
        if (Objects.requireNonNull(first) == Objects.requireNonNull(second)) {
            return 0;
        }
        int commonLength = Math.min(first.length(), second.length());
        for (int i = 0; i < commonLength; i++) {
            int fi = first.intAt(i);
            int si = second.intAt(i);
            if (fi != si) {
                return Integer.compare(fi, si);
            }
        }
        return Integer.compare(first.length(), second.length());
    }

    /**
     * Returns the first element of this sequence.
     *
     * @throws NoSuchElementException if the sequence is empty
     */
    default int getFirst() {
        if (length() == 0) {
            throw new NoSuchElementException("Sequence is empty");
        }
        return intAt(0);
    }

    /**
     * Returns the last element of this sequence.
     *
     * @throws NoSuchElementException if the sequence is empty
     */
    default int getLast() {
        if (length() == 0) {
            throw new NoSuchElementException("Sequence is empty");
        }
        return intAt(length() - 1);
    }

    /** Returns whether this sequence contains no elements. */
    default boolean isEmpty() {
        return length() == 0;
    }

    /** Copies all values in this sequence into {@code dest} starting at {@code destOffset}. */
    default void copyTo(int[] dest, int destOffset) {
        copyTo(0, dest, destOffset, length());
    }

    /**
     * Copies {@code count} values from the start of this sequence into {@code dest} starting at
     * {@code destOffset}.
     */
    default void copyTo(int[] dest, int destOffset, int count) {
        copyTo(0, dest, destOffset, count);
    }

    /**
     * Copies {@code count} values from this sequence, starting at {@code srcOffset}, into {@code
     * dest} starting at {@code destOffset}.
     */
    default void copyTo(int srcOffset, int[] dest, int destOffset, int count) {
        Objects.requireNonNull(dest, "dest");
        int sequenceLength = length();
        if (srcOffset < 0 || srcOffset > sequenceLength) {
            throw new IndexOutOfBoundsException("srcOffset: " + srcOffset);
        }
        if (destOffset < 0 || destOffset > dest.length) {
            throw new IndexOutOfBoundsException("destOffset: " + destOffset);
        }
        if (count < 0 || count > sequenceLength - srcOffset) {
            throw new IndexOutOfBoundsException(
                    "count: "
                            + count
                            + ", available from srcOffset "
                            + srcOffset
                            + " is "
                            + (sequenceLength - srcOffset));
        }
        if (count > dest.length - destOffset) {
            throw new IndexOutOfBoundsException(
                    "Destination too small: need " + count + " at offset " + destOffset);
        }
        for (int i = 0; i < count; i++) {
            dest[destOffset + i] = intAt(srcOffset + i);
        }
    }

    /** Applies {@code action} to each element of this sequence, in order. */
    default void forEachInt(IntConsumer action) {
        Objects.requireNonNull(action, "action");
        int sequenceLength = length();
        for (int i = 0; i < sequenceLength; i++) {
            action.accept(intAt(i));
        }
    }

    /** Returns whether this sequence starts with the given prefix. */
    default boolean startsWith(IntSequence prefix) {
        IntSequence nonNullPrefix = Objects.requireNonNull(prefix, "prefix");
        int thisLength = length();
        int prefixLength = nonNullPrefix.length();
        if (prefixLength > thisLength) {
            return false;
        }
        for (int i = 0; i < prefixLength; i++) {
            if (intAt(i) != nonNullPrefix.intAt(i)) {
                return false;
            }
        }
        return true;
    }

    /** Returns whether this sequence ends with the given suffix. */
    default boolean endsWith(IntSequence suffix) {
        IntSequence nonNullSuffix = Objects.requireNonNull(suffix, "suffix");
        int thisLength = length();
        int suffixLength = nonNullSuffix.length();
        if (suffixLength > thisLength) {
            return false;
        }
        int start = thisLength - suffixLength;
        for (int i = 0; i < suffixLength; i++) {
            if (intAt(start + i) != nonNullSuffix.intAt(i)) {
                return false;
            }
        }
        return true;
    }

    /** Concatenates this sequence with another sequence. */
    default IntSequence concat(IntSequence other) {
        IntSequence nonNullOther = Objects.requireNonNull(other, "other");
        if (isEmpty()) {
            return nonNullOther;
        }
        if (nonNullOther.isEmpty()) {
            return this;
        }
        int thisLength = length();
        int[] merged = new int[thisLength + nonNullOther.length()];
        copyTo(merged, 0);
        nonNullOther.copyTo(merged, thisLength);
        return wrap(merged);
    }

    /**
     * Returns the index of the first occurrence of {@code value} in this sequence, or -1 if not
     * found.
     */
    default int indexOf(int value) {
        int len = length();
        for (int i = 0; i < len; i++) {
            if (intAt(i) == value) {
                return i;
            }
        }
        return -1;
    }

    /**
     * Returns the index of the first occurrence of {@code value} starting at or after {@code
     * fromIndex}, or -1 if not found.
     */
    default int indexOf(int value, int fromIndex) {
        int len = length();
        for (int i = Math.max(0, fromIndex); i < len; i++) {
            if (intAt(i) == value) {
                return i;
            }
        }
        return -1;
    }

    /**
     * Returns the index of the last occurrence of {@code value} in this sequence, or -1 if not
     * found.
     */
    default int lastIndexOf(int value) {
        int len = length();
        for (int i = len - 1; i >= 0; i--) {
            if (intAt(i) == value) {
                return i;
            }
        }
        return -1;
    }

    /**
     * Returns the index of the last occurrence of {@code value} in this sequence, searching
     * backward from {@code fromIndex}, or -1 if not found.
     */
    default int lastIndexOf(int value, int fromIndex) {
        int len = length();
        for (int i = Math.min(fromIndex, len - 1); i >= 0; i--) {
            if (intAt(i) == value) {
                return i;
            }
        }
        return -1;
    }

    /**
     * A builder interface for creating IntSequence instances.
     *
     * <p>Use {@link #snapshot()} for a fixed-length view, {@link #asSequenceView()} for a live view
     * that reflects subsequent additions, and {@link #build()} for a stable copied sequence.
     */
    interface Builder {
        /** Returns the number of elements added so far. */
        int size();

        /** Returns whether no elements have been added yet. */
        default boolean isEmpty() {
            return size() == 0;
        }

        /**
         * Ensures that the builder can hold at least the specified number of elements.
         *
         * @param minCapacity the minimum capacity needed
         */
        void ensureCapacity(int minCapacity);

        /** Appends {@code value} and returns this builder. */
        Builder add(int value);

        /** Returns a stable copied sequence of the elements added so far. */
        IntSequence build();

        /**
         * Returns a fixed-length unmodifiable view over the current builder contents.
         *
         * <p>The returned sequence does not grow as new elements are added to this builder.
         * However, it shares backing storage and may reflect in-place mutations of already-visible
         * elements.
         */
        IntSequence snapshot();

        /**
         * Returns a live unmodifiable view over this builder.
         *
         * <p>The returned sequence reflects subsequent additions to this builder and shares backing
         * storage, so in-place element mutations are also visible.
         */
        IntSequence asSequenceView();

        /** Appends all elements of {@code elems} and returns this builder. */
        default Builder addAll(IntSequence elems) {
            IntSequence nonNullElems = Objects.requireNonNull(elems, "elems");
            ensureCapacity(this.size() + nonNullElems.length());
            int size = nonNullElems.length();
            for (int i = 0; i < size; ++i) {
                add(nonNullElems.intAt(i));
            }
            return this;
        }

        /**
         * Appends all elements of another builder, read as a fixed-size snapshot at call time, and
         * returns this builder.
         */
        default Builder addAll(Builder elems) {
            return addAll(Objects.requireNonNull(elems, "elems").snapshot());
        }
    }

    /** Returns a new builder with default initial capacity. */
    static Builder newBuilder() {
        return ImplAccessor.newBuilder();
    }

    /**
     * Returns a new builder with the given initial capacity.
     *
     * @throws IllegalArgumentException if {@code initialCapacity} is negative
     */
    static Builder newBuilder(int initialCapacity) {
        return ImplAccessor.newBuilder(initialCapacity);
    }

    /**
     * Formats this sequence as {@code prefix} + elements joined by {@code delimiter} + {@code
     * suffix}.
     */
    default String toString(CharSequence delimiter, CharSequence prefix, CharSequence suffix) {
        StringJoiner joiner = new StringJoiner(delimiter, prefix, suffix);
        int length = length();
        for (int i = 0; i < length; ++i) {
            joiner.add(Integer.toString(intAt(i)));
        }
        return joiner.toString();
    }
}
