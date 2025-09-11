package com.llm4j.api;

/**
 * Represents basic information about a tensor in a machine learning model.
 * This interface provides essential metadata about tensors including their
 * names, data types, and dimensional structure.
 */
public interface BaseTensorInfo {

    /**
     * Returns the identifier name of the tensor.
     * This name is typically used to reference the tensor within the model
     * or computational graph.
     *
     * @return The string name identifier of the tensor
     */
    String name();

    /**
     * Returns the base data type of the tensor's elements.
     * This defines the numerical format used to store the tensor's values
     * (e.g., float32, int64, etc.).
     *
     * @return The BaseType enum value representing the tensor's data type
     */
    BaseType type();

    /**
     * Returns the dimensional structure of the tensor as an array of lengths.
     * For example:
     * - A vector would have shape [n]
     * - A matrix would have shape [rows, cols]
     * - A 3D tensor would have shape [depth, rows, cols]
     *
     * @return An array of long values representing the size of each dimension
     */
    long[] shape();
}