package com.llm4j.api;

import java.util.stream.LongStream;

/**
 * Defines the base data type interface for tensor elements in a machine learning model.
 * This interface provides methods to determine the memory characteristics and quantization
 * status of different data types used in tensor operations.
 */
public interface BaseType {

    /**
     * Indicates whether this data type represents quantized values.
     * Quantization is a technique that reduces the memory footprint and computational
     * requirements by representing numbers with lower precision.
     *
     * @return true if the type represents quantized values, false otherwise
     */
    boolean isQuantized();

    /**
     * Calculates the total number of bytes required to store the specified number of elements
     * of this type.
     *
     * @param numberOfElements the number of elements to calculate storage for
     * @return the total number of bytes required for storage
     */
    long byteSizeFor(long numberOfElements);

    /**
     * Calculates the total number of bytes required to store a tensor with the specified
     * dimensions using this data type. This is a convenience method that computes the
     * total number of elements from the dimensions and then calculates the byte size.
     *
     * @param dims variable number of dimension sizes
     * @return the total number of bytes required for storage
     */
    default long byteSizeForShape(long... dims) {
        return byteSizeFor(numberOfElements(dims));
    }

    /**
     * Utility method to calculate the total number of elements in a tensor given its dimensions.
     * This method multiplies all dimension sizes together to get the total element count.
     *
     * @param dims variable number of dimension sizes
     * @return the total number of elements in a tensor with the specified dimensions
     */
    static long numberOfElements(long... dims) {
        assert LongStream.of(dims).allMatch(d -> d > 0);
        return LongStream.of(dims).reduce(1, Math::multiplyExact);
    }
}