package com.llm4j.span;

/**
 * A generic interface representing a contiguous sequence of elements.
 * Spans provide efficient access to sequences of data by acting as a window over
 * the underlying storage without copying the data. This is particularly useful
 * for working with large arrays or memory regions.
 *
 * @param <T> The type of elements contained in the span
 */
public interface Span<T> {

    /**
     * Returns the number of elements in this span.
     *
     * @return The total number of elements accessible through this span
     */
    long size();

    /**
     * Returns the total size in bytes of the data viewed by this span.
     * This may be different from size() * sizeof(T) if the data is stored
     * in a compressed or packed format.
     *
     * @return The total number of bytes occupied by the span's data
     */
    long sizeInBytes();

    /**
     * Creates a new span representing a subsection of this span.
     * The new span shares the same underlying data but provides a view
     * over a different range of elements.
     *
     * @param spanStartIndex The starting index in this span where the slice should begin
     * @param spanLength     The length of the slice to create
     * @return A new span representing the specified slice
     * @throws IndexOutOfBoundsException if spanStartIndex is negative or if
     *                                  spanStartIndex + spanLength exceeds the size of this span
     */
    Span<T> slice(long spanStartIndex, long spanLength);

    /**
     * Creates a new span representing a subsection of this span,
     * starting at the specified index and continuing to the end of the span.
     * This is a convenience method that calculates the appropriate length
     * based on the starting index.
     *
     * @param spanStartIndex The starting index in this span where the slice should begin
     * @return A new span representing the specified slice
     * @throws IndexOutOfBoundsException if spanStartIndex is negative or
     *                                  exceeds the size of this span
     */
    default Span<T> slice(long spanStartIndex) {
        return slice(spanStartIndex, size() - spanStartIndex);
    }
}