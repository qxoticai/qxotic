package ai.qxotic.tokenizers;

import ai.qxotic.tokenizers.impl.ImplAccessor;

import java.util.*;
import java.util.stream.IntStream;
import java.util.stream.StreamSupport;

/**
 * A sequence of integer values providing uniform, read-only access to different kinds of integer collections.
 * This interface combines the functionality of both primitive int arrays and Integer collections while
 * maintaining efficient memory usage and performance characteristics.
 * <p>
 * The interface provides:
 * <ul>
 *   <li>Read-only access to sequence elements</li>
 *   <li>Conversion methods to arrays and collections</li>
 *   <li>Stream processing capabilities</li>
 *   <li>Builder pattern for creating sequences</li>
 *   <li>Utility methods for sequence comparison and manipulation</li>
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
     * @param start the start index, inclusive
     * @param end   the end index, exclusive
     * @return a new sequence containing the specified range of elements
     * @throws IndexOutOfBoundsException if start or end are out of bounds
     * @throws IllegalArgumentException  if start is greater than end
     */
    IntSequence subSequence(int start, int end);

    /**
     * Converts this sequence to a new integer array containing all elements.
     *
     * @return a new array containing all elements in this sequence
     */
    default int[] toArray() {
        int length = length();
        int[] array = new int[length];
        for (int i = 0; i < length; i++) {
            array[i] = intAt(i);
        }
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
                        Spliterator.ORDERED | Spliterator.SIZED | Spliterator.SUBSIZED
                ),
                false
        );
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
        return wrap(values);
    }

    /**
     * Creates an IntSequence from the given list of Integers.
     *
     * @param integerList the list of Integers to wrap
     * @return a new IntSequence containing the elements from the list
     * @throws NullPointerException if integerList is null
     */
    static IntSequence wrap(List<Integer> integerList) {
        return ImplAccessor.wrap(integerList);
    }

    /**
     * Creates an IntSequence from the given array.
     *
     * @param array the array to wrap
     * @return a new IntSequence containing the elements from the array
     * @throws NullPointerException if array is null
     */
    static IntSequence wrap(int[] array) {
        if (array.length == 0) {
            return empty(); // cannot grow
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

    /**
     * A builder interface for creating IntSequence instances.
     * The builder itself implements IntSequence, allowing for efficient
     * building operations while maintaining read access to the current sequence.
     */
    interface Builder extends IntSequence {
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
         * Adds all elements from the given sequence to this builder.
         *
         * @param elems the sequence of elements to add
         * @return this builder instance
         * @throws NullPointerException if elems is null
         */
        default Builder addAll(IntSequence elems) {
            ensureCapacity(this.length() + elems.length());
            int size = elems.length();
            for (int i = 0; i < size; ++i) {
                add(elems.intAt(i));
            }
            return this;
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
     * Compares two sequences for content equality.
     *
     * @param first  the first sequence
     * @param second the second sequence
     * @return true if both sequences have the same length and contain the same elements in the same order
     * @throws NullPointerException if either sequence is null
     */
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

    /**
     * Compares two sequences lexicographically.
     *
     * @param first  the first sequence
     * @param second the second sequence
     * @return negative if first < second, zero if equal, positive if first > second
     * @throws NullPointerException if either sequence is null
     */
    static int compare(IntSequence first, IntSequence second) {
        if (Objects.requireNonNull(first) == Objects.requireNonNull(second)) {
            return 0;
        }
        int commonLength = Math.min(first.length(), second.length());
        for (int i = 0; i < commonLength; i++) {
            int fi = first.intAt(i);
            int si = second.intAt(i);
            if (fi != si) {
                return fi - si;
            }
        }
        return first.length() - second.length();
    }

    /**
     * Returns a string representation of this sequence using the specified delimiter and enclosing symbols.
     *
     * @param delimiter the separator between elements
     * @param prefix    the prefix for the string representation
     * @param suffix    the suffix for the string representation
     * @return a string representation of this sequence
     * @throws NullPointerException if any parameter is null
     */
    default String toString(CharSequence delimiter, CharSequence prefix, CharSequence suffix) {
        StringBuilder sb = new StringBuilder(prefix);
        if (!isEmpty()) {
            sb.append(intAt(0));
            for (int i = 1; i < length(); ++i) {
                sb.append(delimiter);
                sb.append(intAt(i));
            }
        }
        return sb.append(suffix).toString();
    }
}