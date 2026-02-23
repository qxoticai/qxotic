package com.qxotic.span;

/**
 * Represents a view or slice over a contiguous sequence of float values. This interface extends the
 * generic Span interface specifically for Float data types, providing specialized methods for
 * working with floating-point number sequences. The span acts as a window over the underlying data
 * without copying it.
 */
public interface FloatSpan extends Span<Float> {

    /**
     * Creates a new FloatSpan representing a subsection of this span. The new span shares the same
     * underlying data but provides a view over a different range of elements.
     *
     * @param fromIndex The starting index in this span where the slice should begin
     * @param spanLength The length of the slice to create
     * @return A new FloatSpan representing the specified slice
     * @throws IndexOutOfBoundsException if fromIndex is negative or if fromIndex + spanLength
     *     exceeds the size of this span
     */
    @Override
    FloatSpan slice(long fromIndex, long spanLength);

    /**
     * Creates a new FloatSpan representing a subsection of this span, starting at the specified
     * index and continuing to the end of the span. This is a convenience method that calculates the
     * appropriate length based on the starting index.
     *
     * @param fromIndex The starting index in this span where the slice should begin
     * @return A new FloatSpan representing the specified slice
     * @throws IndexOutOfBoundsException if fromIndex is negative or exceeds the size of this span
     */
    @Override
    default FloatSpan slice(long fromIndex) {
        return slice(fromIndex, size() - fromIndex);
    }
}
