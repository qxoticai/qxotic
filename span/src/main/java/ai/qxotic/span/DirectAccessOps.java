package ai.qxotic.span;

/**
 * Interface providing direct element-wise access to float spans. This represents a low-level,
 * potentially less efficient interface that should only be used for debugging, testing,
 * or when direct access to tensor elements is absolutely necessary.
 *
 * @param <S> The specific type of FloatSpan to operate on
 */
public interface DirectAccessOps<S extends FloatSpan> {
    /**
     * Retrieves a single element from the span at the specified index.
     * This operation may be inefficient for some implementations (e.g., GPU-backed spans).
     *
     * @param span  The span to read from
     * @param index The index of the element to retrieve
     * @return The float value at the specified index
     * @throws IndexOutOfBoundsException if index is negative or >= span.size()
     */
    float getElementAt(S span, long index);

    /**
     * Sets a single element in the span at the specified index.
     * This operation may be inefficient for some implementations (e.g., GPU-backed spans).
     *
     * @param span  The span to write to
     * @param index The index of the element to set
     * @param value The float value to write
     * @throws IndexOutOfBoundsException if index is negative or >= span.size()
     */
    void setElementAt(S span, long index, float value);

    /**
     * Reduces a range within a span to a single value using a binary operator.
     * The reduction is performed sequentially from left to right.
     *
     * @param span      The span to reduce
     * @param fromIndex The initial index of the range (inclusive)
     * @param toIndex   The final index of the range (exclusive)
     * @param seed      The initial value for the reduction
     * @param reducer   The binary operator to use for reduction
     * @return The final reduced value
     * @throws IllegalArgumentException if fromIndex > toIndex or if the range is out of bounds
     */
    default float fold(S span, long fromIndex, long toIndex, float seed, FloatBinaryOperator reducer) {
        checkArg(fromIndex <= toIndex, "fromIndex > toIndex");
        checkArg(0 <= fromIndex && toIndex <= span.size(), "range out-of-bounds");
        float result = seed;
        for (long i = fromIndex; i < toIndex; ++i) {
            result = reducer.apply(result, getElementAt(span, i));
        }
        return result;
    }

    default float fold(S span, float seed, FloatBinaryOperator reducer) {
        return fold(span, 0, span.size(), seed, reducer);
    }

    /**
     * Finds the minimum value in a range within the span.
     *
     * @param span      The span to search
     * @param fromIndex The initial index of the range (inclusive)
     * @param toIndex   The final index of the range (exclusive)
     * @return The minimum value in the specified range
     * @throws IllegalArgumentException if fromIndex > toIndex or if the range is out of bounds
     */
    default float min(S span, long fromIndex, long toIndex) {
        return fold(span, fromIndex, toIndex, Float.POSITIVE_INFINITY, FloatBinaryOperator.MIN);
    }

    /**
     * Finds the minimum value in the entire span.
     *
     * @param span The span to search
     * @return The minimum value in the span
     */
    default float min(S span) {
        return min(span, 0, span.size());
    }

    /**
     * Finds the maximum value in a range within the span.
     *
     * @param span      The span to search
     * @param fromIndex The initial index of the range (inclusive)
     * @param toIndex   The final index of the range (exclusive)
     * @return The maximum value in the specified range
     * @throws IllegalArgumentException if fromIndex > toIndex or if the range is out of bounds
     */
    default float max(S span, long fromIndex, long toIndex) {
        return fold(span, fromIndex, toIndex, Float.NEGATIVE_INFINITY, FloatBinaryOperator.MAX);
    }

    /**
     * Finds the maximum value in the entire span.
     *
     * @param span The span to search
     * @return The maximum value in the span
     */
    default float max(S span) {
        return max(span, 0, span.size());
    }

    /**
     * Computes the sum of all elements in a range within the span.
     *
     * @param span      The span to sum
     * @param fromIndex The initial index of the range (inclusive)
     * @param toIndex   The final index of the range (exclusive)
     * @return The sum of all elements in the specified range
     * @throws IllegalArgumentException if fromIndex > toIndex or if the range is out of bounds
     */
    default float sum(S span, long fromIndex, long toIndex) {
        return fold(span, fromIndex, toIndex, 0f, FloatBinaryOperator.SUM);
    }

    /**
     * Computes the sum of all elements in the span.
     *
     * @param span The span to sum
     * @return The sum of all elements
     */
    default float sum(S span) {
        return sum(span, 0, span.size());
    }

    /**
     * Computes the product of all elements in a range within the span.
     *
     * @param span      The span to multiply
     * @param fromIndex The initial index of the range (inclusive)
     * @param toIndex   The final index of the range (exclusive)
     * @return The product of all elements in the specified range
     * @throws IllegalArgumentException if fromIndex > toIndex or if the range is out of bounds
     */
    default float product(S span, long fromIndex, long toIndex) {
        return fold(span, fromIndex, toIndex, 1f, FloatBinaryOperator.MUL);
    }

    /**
     * Computes the product of all elements in the span.
     *
     * @param span The span to multiply
     * @return The product of all elements
     */
    default float product(S span) {
        return product(span, 0, span.size());
    }

    /**
     * Finds the index of the maximum value in a range within the span.
     * If multiple elements have the maximum value, returns the first occurrence.
     *
     * @param span      The span to search
     * @param fromIndex The initial index of the range (inclusive)
     * @param toIndex   The final index of the range (exclusive)
     * @return The index of the maximum value within the specified range
     * @throws IllegalArgumentException if fromIndex > toIndex or if the range is out of bounds
     */
    default long argMax(S span, long fromIndex, long toIndex) {
        checkArg(fromIndex <= toIndex, "fromIndex > toIndex");
        checkArg(0 <= fromIndex && toIndex <= span.size(), "range out-of-bounds");
        long maxIndex = -1;
        float maxValue = Float.NEGATIVE_INFINITY;
        for (long i = fromIndex; i < toIndex; ++i) {
            float value = getElementAt(span, i);
            if (value > maxValue) {
                maxValue = value;
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    /**
     * Finds the index of the maximum value in the entire span.
     * If multiple elements have the maximum value, returns the first occurrence.
     *
     * @param span The span to search
     * @return The index of the maximum value in the span
     */
    default long argMax(S span) {
        return argMax(span, 0, span.size());
    }

    /**
     * Finds the index of the minimum value in a range within the span.
     * If multiple elements have the minimum value, returns the first occurrence.
     *
     * @param span      The span to search
     * @param fromIndex The initial index of the range (inclusive)
     * @param toIndex   The final index of the range (exclusive)
     * @return The index of the minimum value within the specified range
     * @throws IllegalArgumentException if fromIndex > toIndex or if the range is out of bounds
     */
    default long argMin(S span, long fromIndex, long toIndex) {
        checkArg(fromIndex <= toIndex, "fromIndex > toIndex");
        checkArg(0 <= fromIndex && toIndex <= span.size(), "range out-of-bounds");
        long minIndex = -1;
        float minValue = Float.POSITIVE_INFINITY;
        for (long i = fromIndex; i < toIndex; ++i) {
            float value = getElementAt(span, i);
            if (value < minValue) {
                minValue = value;
                minIndex = i;
            }
        }
        return minIndex;
    }

    /**
     * Finds the index of the minimum value in the entire span.
     * If multiple elements have the minimum value, returns the first occurrence.
     *
     * @param span The span to search
     * @return The index of the minimum value in the span
     */
    default long argMin(S span) {
        return argMin(span, 0, span.size());
    }

    private static void checkArg(boolean condition, String message) {
        if (!condition) {
            throw new IllegalArgumentException(message);
        }
    }
}