package ai.qxotic.span;

import java.util.stream.IntStream;

/**
 * Specialization of MatrixView for floating-point values, providing efficient access to 2D matrices
 * of float values stored in row-major order. This interface includes several factory methods for
 * creating matrix views over float spans with different configurations.
 */
public interface FloatMatrixView extends MatrixView<Float> {

    /**
     * Returns a view of a single row as a FloatSpan.
     *
     * @param rowIndex The index of the row to view
     * @return A FloatSpan representing the specified row
     * @throws IndexOutOfBoundsException if rowIndex is out of bounds
     */
    @Override
    FloatSpan row(long rowIndex);

    /**
     * Returns the underlying FloatSpan that contains all the matrix data.
     *
     * @return The FloatSpan containing the matrix data
     */
    @Override
    FloatSpan innerSpan();

    /**
     * Creates a matrix view over a float span with full control over layout parameters. This method
     * allows for creation of views with custom row stride and optional row caching.
     *
     * @param span The underlying float span containing the data
     * @param startOffset Starting offset in the span for this matrix
     * @param rows Number of rows in the matrix
     * @param cols Number of columns in the matrix
     * @param rowStride Number of elements between starts of consecutive rows
     * @param preComputeRows If true, pre-computes and caches row spans for faster access
     * @return A new FloatMatrixView with the specified parameters
     * @throws IllegalArgumentException if the parameters specify an invalid matrix configuration
     */
    static FloatMatrixView asMatrix(
            FloatSpan span,
            long startOffset,
            long rows,
            long cols,
            long rowStride,
            boolean preComputeRows) {
        return new FloatMatrixView() {
            final FloatSpan[] rowsSpans =
                    preComputeRows
                            ? IntStream.range(0, Math.toIntExact(rows))
                                    .mapToObj(ri -> span.slice(rowOffset(ri), cols))
                                    .toArray(FloatSpan[]::new)
                            : null;

            @Override
            public FloatSpan row(long rowIndex) {
                assert 0 <= rowIndex && rowIndex < rows;
                return rowsSpans != null
                        ? rowsSpans[(int) rowIndex]
                        : span.slice(rowOffset(rowIndex), cols);
            }

            @Override
            public FloatSpan innerSpan() {
                return span;
            }

            @Override
            public long rows() {
                return rows;
            }

            @Override
            public long cols() {
                return cols;
            }

            public long size() {
                return rows() * cols();
            }

            @Override
            public long rowStride() {
                return rowStride;
            }

            @Override
            public long startOffset() {
                return startOffset;
            }
        };
    }

    /**
     * Creates a matrix view over a float span without row caching.
     *
     * @param span The underlying float span containing the data
     * @param startOffset Starting offset in the span for this matrix
     * @param rows Number of rows in the matrix
     * @param cols Number of columns in the matrix
     * @param rowStride Number of elements between starts of consecutive rows
     * @return A new FloatMatrixView with the specified parameters
     */
    static FloatMatrixView asMatrix(
            FloatSpan span, long startOffset, long rows, long cols, long rowStride) {
        return asMatrix(span, startOffset, rows, cols, rowStride, false);
    }

    /**
     * Creates a matrix view over a float span with automatic row stride calculation. The row stride
     * is set equal to the number of columns, assuming dense packing.
     *
     * @param span The underlying float span containing the data
     * @param startOffset Starting offset in the span for this matrix
     * @param rows Number of rows in the matrix
     * @param cols Number of columns in the matrix
     * @return A new FloatMatrixView with the specified parameters
     * @throws IllegalArgumentException if startOffset is out of bounds or the span is too small
     */
    static FloatMatrixView asMatrix(FloatSpan span, long startOffset, long rows, long cols) {
        if (!(0 <= startOffset && startOffset < span.size())) {
            throw new IllegalArgumentException("offset out-of-bounds");
        }
        if (!(rows * cols <= span.size() - startOffset)) {
            throw new IllegalArgumentException("span size < rows * cols");
        }
        return asMatrix(span, startOffset, rows, cols, cols);
    }

    /**
     * Creates a matrix view starting at the beginning of a float span.
     *
     * @param span The underlying float span containing the data
     * @param rows Number of rows in the matrix
     * @param cols Number of columns in the matrix
     * @return A new FloatMatrixView with the specified dimensions
     */
    static FloatMatrixView asMatrix(FloatSpan span, long rows, long cols) {
        return asMatrix(span, 0, rows, cols, cols);
    }

    /**
     * Creates a cached matrix view optimized for batch processing. This method pre-computes row
     * spans for efficient access and ensures the span size is compatible with the requested batch
     * size.
     *
     * @param span The underlying float span containing the data
     * @param batchSize The size of each batch (number of rows)
     * @return A new FloatMatrixView with pre-computed row spans
     * @throws IllegalArgumentException if span size is not a multiple of batchSize
     */
    static FloatMatrixView inBatchesCached(FloatSpan span, long batchSize) {
        if (span.size() % batchSize != 0) {
            throw new IllegalArgumentException("span size is not a multiple of batchSize");
        }
        long cols = span.size() / batchSize;
        return asMatrix(span, 0, batchSize, cols, cols, true);
    }
}
