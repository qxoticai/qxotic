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
import java.util.function.IntConsumer;
import java.util.stream.IntStream;
import java.util.stream.StreamSupport;

/**
 * A sequence of integer values providing uniform, read-only access to different kinds of integer
 * collections. This interface combines the functionality of both primitive int arrays and Integer
 * collections while maintaining efficient memory usage and performance characteristics.
 *
 * <p>The interface provides:
 *
 * <ul>
 *   <li>Read-only access to sequence elements
 *   <li>Conversion methods to arrays and collections
 *   <li>Stream processing capabilities
 *   <li>Builder pattern for creating sequences
 *   <li>Core operations for composition and partial copying
 * </ul>
 */
public interface IntSequence extends Iterable<Integer>, Comparable<IntSequence> {

    /**
     * Returns an empty IntSequence.
     *
     * @return an empty sequence
     */
    static IntSequence empty() {
        return ImplAccessor.empty();
    }

    /**
     * Returns the integer at the specified position in this sequence.
     *
     * @param index index of the element to return
     * @return the element at the specified position in this sequence
     * @throws IndexOutOfBoundsException if the index is out of range
     */
    int intAt(int index);

    /**
     * Returns the length of this sequence.
     *
     * @return the number of elements in this sequence
     */
    int length();

    /**
     * Returns a new sequence that is a subsequence of this sequence.
     *
     * @param startInclusive the startInclusive index, inclusive
     * @param endExclusive the endExclusive index, exclusive
     * @return a new sequence containing the specified range of elements
     * @throws IndexOutOfBoundsException if startInclusive or endExclusive are out of bounds
     */
    IntSequence subSequence(int startInclusive, int endExclusive);

    /**
     * Converts this sequence to a new integer array containing all elements.
     *
     * @return a new array containing all elements in this sequence
     */
    default int[] toArray() {
        int length = length();
        int[] array = new int[length];
        copyTo(0, array, 0, length);
        return array;
    }

    /**
     * Converts this sequence to a new List containing all elements boxed as Integers.
     *
     * @return a new ArrayList containing all elements in this sequence
     */
    default List<Integer> toList() {
        int length = length();
        List<Integer> list = new ArrayList<>(length);
        for (int i = 0; i < length; i++) {
            list.add(intAt(i));
        }
        return list;
    }

    /**
     * Returns a primitive iterator over the integers in this sequence.
     *
     * @return a PrimitiveIterator.OfInt instance for this sequence
     */
    @Override
    default PrimitiveIterator.OfInt iterator() {
        return new PrimitiveIterator.OfInt() {
            private int index = 0;

            @Override
            public boolean hasNext() {
                return index < length();
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

    /**
     * Creates an IntStream to process the elements of this sequence.
     *
     * @return an IntStream containing the elements of this sequence
     */
    default IntStream stream() {
        return StreamSupport.intStream(
                Spliterators.spliterator(
                        iterator(),
                        length(),
                        Spliterator.ORDERED | Spliterator.SIZED | Spliterator.SUBSIZED),
                false);
    }

    /**
     * Creates an IntSequence from the given values.
     *
     * @param values the values to include in the sequence
     * @return a new IntSequence containing the specified values
     */
    static IntSequence of(int... values) {
        if (values.length == 0) {
            return empty();
        }
        return copyOf(values);
    }

    /**
     * Creates an IntSequence from the given list of Integers.
     *
     * <p>The returned sequence is unmodifiable through this API, but may reflect subsequent
     * mutations to the wrapped list.
     *
     * @param integerList the list of Integers to wrap
     * @return a new IntSequence containing the elements from the list
     */
    static IntSequence wrap(List<Integer> integerList) {
        return ImplAccessor.wrap(integerList);
    }

    /**
     * Creates an IntSequence from the given array.
     *
     * <p>The returned sequence is unmodifiable through this API, but may reflect subsequent
     * mutations to the wrapped array.
     *
     * @param array the array to wrap
     * @return a new IntSequence containing the elements from the array
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

    /**
     * Returns the first integer in this sequence.
     *
     * @return the first element
     * @throws NoSuchElementException if the sequence is empty
     */
    default int getFirst() {
        if (length() == 0) {
            throw new NoSuchElementException("Sequence is empty");
        }
        return intAt(0);
    }

    /**
     * Returns the last integer in this sequence.
     *
     * @return the last element
     * @throws NoSuchElementException if the sequence is empty
     */
    default int getLast() {
        if (length() == 0) {
            throw new NoSuchElementException("Sequence is empty");
        }
        return intAt(length() - 1);
    }

    /**
     * Returns whether this sequence is empty.
     *
     * @return true if this sequence contains no elements
     */
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

    /** Performs the given action for each int value in this sequence. */
    default void forEachInt(IntConsumer action) {
        Objects.requireNonNull(action, "action");
        int sequenceLength = length();
        for (int i = 0; i < sequenceLength; i++) {
            action.accept(intAt(i));
        }
    }

    /** Creates an IntStream containing this sequence values. */
    default IntStream toIntStream() {
        return stream();
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
     * A builder interface for creating IntSequence instances.
     *
     * <p>Use {@link #snapshot()} for a fixed-length view, {@link #asSequenceView()} for a live view
     * that reflects subsequent additions, and {@link #build()} for a stable copied sequence.
     */
    interface Builder {
        /** Returns the current number of elements stored in this builder. */
        int size();

        /** Returns whether this builder currently has no elements. */
        default boolean isEmpty() {
            return size() == 0;
        }

        /**
         * Ensures that the builder can hold at least the specified number of elements.
         *
         * @param minCapacity the minimum capacity needed
         */
        void ensureCapacity(int minCapacity);

        /**
         * Adds a single value to the sequence being built.
         *
         * @param value the value to add
         * @return this builder instance
         */
        Builder add(int value);

        /**
         * Builds and returns the final IntSequence.
         *
         * @return a new IntSequence containing all added elements
         */
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

        /**
         * Adds all elements from the given sequence to this builder.
         *
         * @param elems the sequence of elements to add
         * @return this builder instance
         */
        default Builder addAll(IntSequence elems) {
            IntSequence nonNullElems = Objects.requireNonNull(elems, "elems");
            ensureCapacity(this.size() + nonNullElems.length());
            int size = nonNullElems.length();
            for (int i = 0; i < size; ++i) {
                add(nonNullElems.intAt(i));
            }
            return this;
        }

        /** Adds all elements from another builder. */
        default Builder addAll(Builder elems) {
            return addAll(Objects.requireNonNull(elems, "elems").asSequenceView());
        }
    }

    /**
     * Creates a new Builder instance with default initial capacity.
     *
     * @return a new Builder instance
     */
    static Builder newBuilder() {
        return ImplAccessor.newBuilder();
    }

    /**
     * Creates a new Builder instance with the specified initial capacity.
     *
     * @param initialCapacity the initial capacity of the builder
     * @return a new Builder instance
     * @throws IllegalArgumentException if initialCapacity is negative
     */
    static Builder newBuilder(int initialCapacity) {
        return ImplAccessor.newBuilder(initialCapacity);
    }

    /**
     * Returns a string representation of this sequence using the specified delimiter and enclosing
     * symbols.
     *
     * @param delimiter the separator between elements
     * @param prefix the prefix for the string representation
     * @param suffix the suffix for the string representation
     * @return a string representation of this sequence
     */
    default String toString(CharSequence delimiter, CharSequence prefix, CharSequence suffix) {
        StringBuilder sb = new StringBuilder(prefix);
        int length = length();
        if (length > 0) {
            sb.append(intAt(0));
            for (int i = 1; i < length; ++i) {
                sb.append(delimiter);
                sb.append(intAt(i));
            }
        }
        return sb.append(suffix).toString();
    }
}
