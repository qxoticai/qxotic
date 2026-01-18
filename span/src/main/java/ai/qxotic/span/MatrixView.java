package ai.qxotic.span;

/**
 * Represents a 2D view of a linear span with row-major layout. This interface provides methods to
 * access and manipulate a 2-dimensional matrix structure that is stored in a linear memory layout.
 * The row-major layout means that elements in the same row are stored contiguously.
 *
 * @param <T> The type of elements contained in the matrix
 */
interface MatrixView<T> {
    /**
     * Returns the number of rows in the matrix. This represents the first dimension of the 2D
     * structure.
     *
     * @return The total number of rows in the matrix
     */
    long rows();

    /**
     * Returns the number of columns in the matrix. This represents the second dimension of the 2D
     * structure.
     *
     * @return The total number of columns in the matrix
     */
    long cols();

    /**
     * Returns the row stride - number of elements between the start of consecutive rows. The stride
     * may be larger than the number of columns in cases where there is padding or the matrix is a
     * view into a larger matrix.
     *
     * @return The number of elements between the start of consecutive rows
     */
    long rowStride();

    /**
     * Creates a view of a single row as a span. This provides direct access to elements in the
     * specified row without copying the data.
     *
     * @param rowIndex The index of the row to view
     * @return A span representing the specified row
     * @throws IndexOutOfBoundsException if rowIndex is negative or >= rows()
     */
    Span<T> row(long rowIndex);

    /**
     * Calculates the offset in the underlying linear storage for a given row. This is used
     * internally to map 2D coordinates to linear memory locations.
     *
     * @param rowIndex The index of the row to calculate the offset for
     * @return The offset in the underlying storage where the row begins
     * @throws IndexOutOfBoundsException if rowIndex is out of bounds (if assertions are enabled)
     */
    default long rowOffset(long rowIndex) {
        if (!(0 <= rowIndex && rowIndex < rows())) {
            throw new IndexOutOfBoundsException();
        }
        return startOffset() + rowIndex * rowStride();
    }

    /**
     * Returns the underlying span that contains all the matrix data. This provides access to the
     * raw linear storage of the matrix.
     *
     * @return The span containing the matrix data
     */
    Span<T> innerSpan();

    /**
     * Returns the starting offset of this matrix view within its inner span. This is useful when
     * the matrix view represents a subsection of a larger data structure.
     *
     * @return The offset where this matrix view begins in its inner span
     */
    long startOffset();
}
