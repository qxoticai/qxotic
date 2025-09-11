package com.llm4j.gguf;

import com.llm4j.gguf.impl.ImplAccessor;
import java.util.Collection;
import java.util.Set;

/**
 * Builder interface for the GGUF format.
 *
 * @see <a href="https://github.com/ggerganov/ggml/blob/master/docs/gguf.md">GGUF format specification</a>
 */
public interface Builder extends Cloneable {
    /**
     * Creates a new {@link Builder} from an existing GGUF instance.
     */
    static Builder newBuilder(GGUF gguf) {
        return ImplAccessor.newBuilder(gguf);
    }

    /**
     * Creates a new empty {@link Builder}.
     */
    static Builder newBuilder() {
        return ImplAccessor.newBuilder();
    }

    /**
     * Builds a GGUF instance with automatic tensor offset computation.
     */
    default GGUF build() {
        return build(true);
    }

    /**
     * Builds a GGUF instance.
     *
     * @param recomputeTensorOffsets if true, tensor offsets will be automatically re-computed,
     *                               packed in the same order and respecting the alignment
     */
    GGUF build(boolean recomputeTensorOffsets);

    /**
     * Creates and returns a copy of this object.
     */
    Builder clone();

    /**
     * Sets the GGUF format version.
     */
    Builder setVersion(int newVersion);

    /**
     * Gets the GGUF format version.
     */
    int getVersion();

    /**
     * Sets the alignment value for tensor data.
     *
     * @throws IllegalArgumentException if alignment is not a power of 2
     */
    default Builder setAlignment(int newAlignment) {
        if (newAlignment < 0 || Integer.bitCount(newAlignment) != 1) {
            throw new IllegalArgumentException("alignment must be a power of 2 but was " + newAlignment);
        }
        return putUnsignedInteger(ImplAccessor.alignmentKey(), newAlignment);
    }

    /**
     * Gets the current alignment value or the default if not set.
     */
    default int getAlignment() {
        if (containsKey(ImplAccessor.alignmentKey())) {
            assert getType(ImplAccessor.alignmentKey()) == MetadataValueType.UINT32;
            return getValue(int.class, ImplAccessor.alignmentKey());
        }
        return ImplAccessor.defaultAlignment();
    }

    /**
     * Adds or updates a tensor.
     */
    Builder putTensor(TensorInfo tensorInfo);

    /**
     * Removes a tensor by name.
     */
    Builder removeTensor(String tensorName);

    /**
     * Checks if a tensor exists by name.
     */
    boolean containsTensor(String tensorName);

    /**
     * Gets tensor information by name.
     */
    TensorInfo getTensor(String tensorName);

    /**
     * Checks if a metadata key exists.
     */
    boolean containsKey(String key);

    /**
     * Gets a metadata value associated with the given key, casting it to the specified target class,
     * or null if the key is not found.
     *
     * @see GGUF#getValue(Class, String)
     */
    <T> T getValue(Class<T> targetClass, String key);

    /**
     * Gets all metadata keys, order is preserved.
     *
     * @return the set of metadata keys
     */
    Set<String> getMetadataKeys();

    /**
     * Gets all tensors, order is preserved.
     *
     * @return the collection of tensor information
     */
    Collection<TensorInfo> getTensors();

    /**
     * Gets the component type for the array value associated with the given key.
     *
     * @param key the key to look up
     * @return the component type, or null if the key doesn't exist or value is not an array
     */
    MetadataValueType getComponentType(String key);

    /**
     * Gets the type of the metadata value associated with the given key.
     *
     * @param key the key to look up
     * @return the metadata value type, or null if the key doesn't exist
     */
    MetadataValueType getType(String key);

    /**
     * Removes a metadata key.
     *
     * @param key the key to remove
     * @return this builder instance
     */
    Builder removeKey(String key);

    /**
     * Sets a String for the given metadata key.
     * Value type: {@link MetadataValueType#STRING}
     *
     * @param key   the key to associate the value with
     * @param value the string value to set
     * @return this builder instance
     */
    Builder putString(String key, String value);

    /**
     * Sets a boolean for the given metadata key.
     * Value type: {@link MetadataValueType#BOOL}
     *
     * @param key   the key to associate the value with
     * @param value the boolean value to set
     * @return this builder instance
     */
    Builder putBoolean(String key, boolean value);

    /**
     * Sets a byte for the given metadata key.
     * Value type: {@link MetadataValueType#INT8}
     *
     * @param key   the key to associate the value with
     * @param value the byte value to set
     * @return this builder instance
     */
    Builder putByte(String key, byte value);

    /**
     * Sets an unsigned byte for the given metadata key.
     * Value type: {@link MetadataValueType#UINT8}
     *
     * @param key   the key to associate the value with
     * @param value the unsigned byte value to set
     * @return this builder instance
     */
    Builder putUnsignedByte(String key, byte value);

    /**
     * Sets a short for the given metadata key.
     * Value type: {@link MetadataValueType#INT16}
     *
     * @param key   the key to associate the value with
     * @param value the short value to set
     * @return this builder instance
     */
    Builder putShort(String key, short value);

    /**
     * Sets an unsigned short for the given metadata key.
     * Value type: {@link MetadataValueType#UINT16}
     *
     * @param key   the key to associate the value with
     * @param value the unsigned short value to set
     * @return this builder instance
     */
    Builder putUnsignedShort(String key, short value);

    /**
     * Sets an integer for the given metadata key.
     * Value type: {@link MetadataValueType#INT32}
     *
     * @param key   the key to associate the value with
     * @param value the integer value to set
     * @return this builder instance
     */
    Builder putInteger(String key, int value);

    /**
     * Sets an unsigned integer for the given metadata key.
     * Value type: {@link MetadataValueType#UINT32}
     *
     * @param key   the key to associate the value with
     * @param value the unsigned integer value to set
     * @return this builder instance
     */
    Builder putUnsignedInteger(String key, int value);

    /**
     * Sets a long for the given metadata key.
     * Value type: {@link MetadataValueType#INT64}
     *
     * @param key   the key to associate the value with
     * @param value the long value to set
     * @return this builder instance
     */
    Builder putLong(String key, long value);

    /**
     * Sets an unsigned long for the given metadata key.
     * Value type: {@link MetadataValueType#UINT64}
     *
     * @param key   the key to associate the value with
     * @param value the unsigned long value to set
     * @return this builder instance
     */
    Builder putUnsignedLong(String key, long value);

    /**
     * Sets a float for the given metadata key.
     * Value type: {@link MetadataValueType#FLOAT32}
     *
     * @param key   the key to associate the value with
     * @param value the float value to set
     * @return this builder instance
     */
    Builder putFloat(String key, float value);

    /**
     * Sets a double for the given metadata key.
     * Value type: {@link MetadataValueType#FLOAT64}
     *
     * @param key   the key to associate the value with
     * @param value the double value to set
     * @return this builder instance
     */
    Builder putDouble(String key, double value);

    /**
     * Sets a boolean array for the given metadata key.
     * Value type: {@link MetadataValueType#ARRAY}
     * Component type: {@link MetadataValueType#BOOL}
     *
     * @param key   the key to associate the array with
     * @param value the boolean array to set
     * @return this builder instance
     */
    Builder putArrayOfBoolean(String key, boolean[] value);

    /**
     * Sets a String array for the given metadata key.
     * Value type: {@link MetadataValueType#ARRAY}
     * Component type: {@link MetadataValueType#STRING}
     *
     * @param key   the key to associate the array with
     * @param value the string array to set
     * @return this builder instance
     */
    Builder putArrayOfString(String key, String[] value);

    /**
     * Sets a byte array for the given metadata key.
     * Value type: {@link MetadataValueType#ARRAY}
     * Component type: {@link MetadataValueType#INT8}
     *
     * @param key   the key to associate the array with
     * @param value the byte array to set
     * @return this builder instance
     */
    Builder putArrayOfByte(String key, byte[] value);

    /**
     * Sets an unsigned byte array for the given metadata key.
     * Value type: {@link MetadataValueType#ARRAY}
     * Component type: {@link MetadataValueType#UINT8}
     *
     * @param key   the key to associate the array with
     * @param value the unsigned byte array to set
     * @return this builder instance
     */
    Builder putArrayOfUnsignedByte(String key, byte[] value);

    /**
     * Sets a short array for the given metadata key.
     * Value type: {@link MetadataValueType#ARRAY}
     * Component type: {@link MetadataValueType#INT16}
     *
     * @param key   the key to associate the array with
     * @param value the short array to set
     * @return this builder instance
     */
    Builder putArrayOfShort(String key, short[] value);

    /**
     * Sets an unsigned short array for the given metadata key.
     * Value type: {@link MetadataValueType#ARRAY}
     * Component type: {@link MetadataValueType#UINT16}
     *
     * @param key   the key to associate the array with
     * @param value the unsigned short array to set
     * @return this builder instance
     */
    Builder putArrayOfUnsignedShort(String key, short[] value);

    /**
     * Sets an integer array for the given metadata key.
     * Value type: {@link MetadataValueType#ARRAY}
     * Component type: {@link MetadataValueType#INT32}
     *
     * @param key   the key to associate the array with
     * @param value the integer array to set
     * @return this builder instance
     */
    Builder putArrayOfInteger(String key, int[] value);

    /**
     * Sets an unsigned integer array for the given metadata key.
     * Value type: {@link MetadataValueType#ARRAY}
     * Component type: {@link MetadataValueType#UINT32}
     *
     * @param key   the key to associate the array with
     * @param value the unsigned integer array to set
     * @return this builder instance
     */
    Builder putArrayOfUnsignedInteger(String key, int[] value);

    /**
     * Sets a long array for the given metadata key.
     * Value type: {@link MetadataValueType#ARRAY}
     * Component type: {@link MetadataValueType#INT64}
     *
     * @param key   the key to associate the array with
     * @param value the long array to set
     * @return this builder instance
     */
    Builder putArrayOfLong(String key, long[] value);

    /**
     * Sets an unsigned long array for the given metadata key.
     * Value type: {@link MetadataValueType#ARRAY}
     * Component type: {@link MetadataValueType#UINT64}
     *
     * @param key   the key to associate the array with
     * @param value the unsigned long array to set
     * @return this builder instance
     */
    Builder putArrayOfUnsignedLong(String key, long[] value);

    /**
     * Sets a float array for the given metadata key.
     * Value type: {@link MetadataValueType#ARRAY}
     * Component type: {@link MetadataValueType#FLOAT32}
     *
     * @param key   the key to associate the array with
     * @param value the float array to set
     * @return this builder instance
     */
    Builder putArrayOfFloat(String key, float[] value);

    /**
     * Sets a double array for the given metadata key.
     * Value type: {@link MetadataValueType#ARRAY}
     * Component type: {@link MetadataValueType#FLOAT64}
     *
     * @param key   the key to associate the array with
     * @param value the double array to set
     * @return this builder instance
     */
    Builder putArrayOfDouble(String key, double[] value);
}
