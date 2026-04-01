package com.qxotic.format.gguf;

import com.qxotic.format.gguf.impl.ImplAccessor;
import java.util.Collection;
import java.util.Set;

/**
 * Builder for creating and modifying GGUF files.
 *
 * <p>This interface provides a fluent API for constructing GGUF instances programmatically. You can
 * create new GGUF files from scratch or modify existing ones by copying from an existing {@link
 * GGUF} instance.
 *
 * <p>The builder supports:
 *
 * <ul>
 *   <li>Adding and modifying metadata key-value pairs with type-safe methods
 *   <li>Adding, removing, and rearranging tensors
 *   <li>Configuring GGUF format version and tensor data alignment
 *   <li>Automatic offset computation for tensor data layout
 * </ul>
 *
 * <p>Example usage - creating a new GGUF file:
 *
 * <pre>{@code
 * GGUF gguf = Builder.newBuilder()
 *     .setVersion(3)
 *     .setAlignment(32)
 *     .putString("general.name", "my-model")
 *     .putInteger("llama.context_length", 4096)
 *     .putFloat("llama.rope.freq_base", 10000.0f)
 *     .putTensor(TensorEntry.create("token_embd.weight", new long[]{4096, 32000}, GGMLType.F32, 0))
 *     .build();
 *
 * GGUF.write(gguf, Path.of("model.gguf"));
 * }</pre>
 *
 * <p>Example usage - modifying an existing GGUF file:
 *
 * <pre>{@code
 * GGUF existing = GGUF.read(Path.of("model.gguf"));
 * GGUF modified = Builder.newBuilder(existing)
 *     .putString("general.description", "Modified model")
 *     .removeKey("deprecated_key")
 *     .build();
 *
 * GGUF.write(modified, Path.of("modified.gguf"));
 * }</pre>
 *
 * @see GGUF
 * @see TensorEntry
 * @see <a href="https://github.com/ggml-org/ggml/blob/master/docs/gguf.md">GGUF format
 *     specification</a>
 */
public interface Builder extends Cloneable {
    /**
     * Creates a new {@link Builder} from an existing GGUF instance.
     *
     * <p>The new builder will contain a deep copy of all metadata and tensor information from the
     * existing GGUF.
     *
     * @param gguf the existing GGUF instance to copy from
     * @return a new builder initialized with the existing GGUF data
     * @throws NullPointerException if gguf is null
     */
    static Builder newBuilder(GGUF gguf) {
        return ImplAccessor.newBuilder(gguf);
    }

    /**
     * Creates a new empty {@link Builder}.
     *
     * @return a new empty builder
     */
    static Builder newBuilder() {
        return ImplAccessor.newBuilder();
    }

    /**
     * Builds a GGUF instance with automatic tensor offset computation.
     *
     * <p>This is equivalent to calling {@code build(true)}, which will recompute all tensor offsets
     * to ensure proper alignment and packing.
     *
     * @return the built GGUF instance
     * @see #build(boolean)
     */
    default GGUF build() {
        return build(true);
    }

    /**
     * Builds a GGUF instance.
     *
     * <p>When {@code recomputeTensorOffsets} is true, the builder will:
     *
     * <ul>
     *   <li>Sort tensors by their current order
     *   <li>Compute new offsets to ensure proper alignment (respecting {@link #getAlignment()})
     *   <li>Pack tensor data contiguously after the metadata section
     * </ul>
     *
     * <p>When false, the original offsets are preserved. This is useful when you want to maintain
     * specific tensor positions or when modifying an existing file without changing its layout.
     *
     * @param recomputeTensorOffsets whether to recompute tensor offsets
     * @return the built GGUF instance
     */
    GGUF build(boolean recomputeTensorOffsets);

    /**
     * Creates and returns a copy of this builder.
     *
     * @return a clone of this builder
     */
    Builder clone();

    /**
     * Sets the GGUF format version.
     *
     * <p>The format version determines how the GGUF file is structured. Version 3 is the current
     * version and supports all features including large file support and all GGML types.
     *
     * @param newVersion the format version to set
     * @return this builder instance
     */
    Builder setVersion(int newVersion);

    /**
     * Gets the GGUF format version.
     *
     * @return the format version
     */
    int getVersion();

    /**
     * Sets the alignment value for tensor data.
     *
     * <p>Alignment ensures that tensor data starts at memory addresses that are multiples of this
     * value. This is important for performance on many architectures. Common values are 32 or 64.
     *
     * <p>The alignment must be a power of 2 (e.g., 1, 2, 4, 8, 16, 32, 64).
     *
     * @param newAlignment the alignment value in bytes, must be a power of 2
     * @return this builder instance
     * @throws IllegalArgumentException if alignment is not a power of 2
     * @see #getAlignment()
     */
    default Builder setAlignment(int newAlignment) {
        if (newAlignment < 0 || Integer.bitCount(newAlignment) != 1) {
            throw new IllegalArgumentException(
                    "alignment must be a power of 2 but was " + newAlignment);
        }
        return putUnsignedInteger(ImplAccessor.alignmentKey(), newAlignment);
    }

    /**
     * Gets the current alignment value or the default if not set.
     *
     * <p>If no alignment has been explicitly set, returns the default alignment value (typically
     * 32).
     *
     * @return the alignment value in bytes
     * @see #setAlignment(int)
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
     *
     * <p>If a tensor with the same name already exists, it will be replaced. The tensor's offset is
     * relative to the start of the tensor data section (after the metadata). Offsets will be
     * recomputed during {@link #build()} unless explicitly disabled.
     *
     * @param tensorEntry the tensor entry to add
     * @return this builder instance
     * @throws NullPointerException if tensorEntry is null
     * @see TensorEntry#create(String, long[], GGMLType, long)
     */
    Builder putTensor(TensorEntry tensorEntry);

    /**
     * Removes a tensor by name.
     *
     * <p>If no tensor with the given name exists, this method does nothing.
     *
     * @param tensorName the name of the tensor to remove
     * @return this builder instance
     */
    Builder removeTensor(String tensorName);

    /**
     * Checks if a tensor exists by name.
     *
     * @param tensorName the tensor name to check
     * @return {@code true} if a tensor with the given name exists
     */
    boolean containsTensor(String tensorName);

    /**
     * Gets tensor information by name.
     *
     * @param tensorName the tensor name to look up
     * @return the tensor entry, or {@code null} if not found
     */
    TensorEntry getTensor(String tensorName);

    /**
     * Checks if a metadata key exists.
     *
     * @param key the metadata key to check
     * @return {@code true} if the key exists
     */
    boolean containsKey(String key);

    /**
     * Gets a metadata value associated with the given key, casting it to the specified target
     * class, or null if the key is not found.
     *
     * <p>See {@link GGUF#getValue(Class, String)} for details on type mapping and examples.
     *
     * @param <T> the target type
     * @param targetClass the class to cast the value to
     * @param key the metadata key
     * @return the value cast to the target class, or {@code null} if the key is not found
     * @see GGUF#getValue(Class, String)
     */
    <T> T getValue(Class<T> targetClass, String key);

    /**
     * Gets all metadata keys.
     *
     * <p>The iteration order of the returned set preserves the insertion order of keys.
     *
     * @return an unmodifiable set of metadata keys
     */
    Set<String> getMetadataKeys();

    /**
     * Gets all tensors.
     *
     * <p>The iteration order of the returned collection preserves the insertion order of tensors.
     *
     * @return an unmodifiable collection of tensor entries
     */
    Collection<TensorEntry> getTensors();

    /**
     * Gets the component type for the array value associated with the given key.
     *
     * <p>This method returns the element type of arrays stored in metadata. For example, if the key
     * maps to a {@code float[]}, this returns {@link MetadataValueType#FLOAT32}.
     *
     * @param key the metadata key
     * @return the component type, or {@code null} if the key is not found
     * @see MetadataValueType#ARRAY
     */
    MetadataValueType getComponentType(String key);

    /**
     * Gets the type of the metadata value associated with the given key.
     *
     * @param key the metadata key
     * @return the value type, or {@code null} if the key is not found
     */
    MetadataValueType getType(String key);

    /**
     * Removes a metadata key.
     *
     * <p>If the key does not exist, this method does nothing.
     *
     * @param key the metadata key to remove
     * @return this builder instance
     */
    Builder removeKey(String key);

    /**
     * Sets a String for the given metadata key. Value type: {@link MetadataValueType#STRING}
     *
     * @param key the metadata key
     * @param value the string value
     * @return this builder instance
     */
    Builder putString(String key, String value);

    /**
     * Sets a boolean for the given metadata key. Value type: {@link MetadataValueType#BOOL}
     *
     * @param key the metadata key
     * @param value the boolean value
     * @return this builder instance
     */
    Builder putBoolean(String key, boolean value);

    /**
     * Sets a byte for the given metadata key. Value type: {@link MetadataValueType#INT8}
     *
     * @param key the metadata key
     * @param value the byte value
     * @return this builder instance
     */
    Builder putByte(String key, byte value);

    /**
     * Sets an unsigned byte for the given metadata key. Value type: {@link MetadataValueType#UINT8}
     *
     * @param key the metadata key
     * @param value the byte value (interpreted as unsigned)
     * @return this builder instance
     */
    Builder putUnsignedByte(String key, byte value);

    /**
     * Sets a short for the given metadata key. Value type: {@link MetadataValueType#INT16}
     *
     * @param key the metadata key
     * @param value the short value
     * @return this builder instance
     */
    Builder putShort(String key, short value);

    /**
     * Sets an unsigned short for the given metadata key. Value type: {@link
     * MetadataValueType#UINT16}
     *
     * @param key the metadata key
     * @param value the short value (interpreted as unsigned)
     * @return this builder instance
     */
    Builder putUnsignedShort(String key, short value);

    /**
     * Sets an integer for the given metadata key. Value type: {@link MetadataValueType#INT32}
     *
     * @param key the metadata key
     * @param value the integer value
     * @return this builder instance
     */
    Builder putInteger(String key, int value);

    /**
     * Sets an unsigned integer for the given metadata key. Value type: {@link
     * MetadataValueType#UINT32}
     *
     * @param key the metadata key
     * @param value the integer value (interpreted as unsigned)
     * @return this builder instance
     */
    Builder putUnsignedInteger(String key, int value);

    /**
     * Sets a long for the given metadata key. Value type: {@link MetadataValueType#INT64}
     *
     * @param key the metadata key
     * @param value the long value
     * @return this builder instance
     */
    Builder putLong(String key, long value);

    /**
     * Sets an unsigned long for the given metadata key. Value type: {@link
     * MetadataValueType#UINT64}
     *
     * @param key the metadata key
     * @param value the long value (interpreted as unsigned)
     * @return this builder instance
     */
    Builder putUnsignedLong(String key, long value);

    /**
     * Sets a float for the given metadata key. Value type: {@link MetadataValueType#FLOAT32}
     *
     * @param key the metadata key
     * @param value the float value
     * @return this builder instance
     */
    Builder putFloat(String key, float value);

    /**
     * Sets a double for the given metadata key. Value type: {@link MetadataValueType#FLOAT64}
     *
     * @param key the metadata key
     * @param value the double value
     * @return this builder instance
     */
    Builder putDouble(String key, double value);

    /**
     * Sets a boolean array for the given metadata key. Value type: {@link MetadataValueType#ARRAY}
     * Component type: {@link MetadataValueType#BOOL}
     *
     * @param key the metadata key
     * @param value the boolean array
     * @return this builder instance
     */
    Builder putArrayOfBoolean(String key, boolean[] value);

    /**
     * Sets a String array for the given metadata key. Value type: {@link MetadataValueType#ARRAY}
     * Component type: {@link MetadataValueType#STRING}
     *
     * @param key the metadata key
     * @param value the string array
     * @return this builder instance
     */
    Builder putArrayOfString(String key, String[] value);

    /**
     * Sets a byte array for the given metadata key. Value type: {@link MetadataValueType#ARRAY}
     * Component type: {@link MetadataValueType#INT8}
     *
     * @param key the metadata key
     * @param value the byte array
     * @return this builder instance
     */
    Builder putArrayOfByte(String key, byte[] value);

    /**
     * Sets an unsigned byte array for the given metadata key. Value type: {@link
     * MetadataValueType#ARRAY} Component type: {@link MetadataValueType#UINT8}
     *
     * @param key the metadata key
     * @param value the byte array (interpreted as unsigned)
     * @return this builder instance
     */
    Builder putArrayOfUnsignedByte(String key, byte[] value);

    /**
     * Sets a short array for the given metadata key. Value type: {@link MetadataValueType#ARRAY}
     * Component type: {@link MetadataValueType#INT16}
     *
     * @param key the metadata key
     * @param value the short array
     * @return this builder instance
     */
    Builder putArrayOfShort(String key, short[] value);

    /**
     * Sets an unsigned short array for the given metadata key. Value type: {@link
     * MetadataValueType#ARRAY} Component type: {@link MetadataValueType#UINT16}
     *
     * @param key the metadata key
     * @param value the short array (interpreted as unsigned)
     * @return this builder instance
     */
    Builder putArrayOfUnsignedShort(String key, short[] value);

    /**
     * Sets an integer array for the given metadata key. Value type: {@link MetadataValueType#ARRAY}
     * Component type: {@link MetadataValueType#INT32}
     *
     * @param key the metadata key
     * @param value the integer array
     * @return this builder instance
     */
    Builder putArrayOfInteger(String key, int[] value);

    /**
     * Sets an unsigned integer array for the given metadata key. Value type: {@link
     * MetadataValueType#ARRAY} Component type: {@link MetadataValueType#UINT32}
     *
     * @param key the metadata key
     * @param value the integer array (interpreted as unsigned)
     * @return this builder instance
     */
    Builder putArrayOfUnsignedInteger(String key, int[] value);

    /**
     * Sets a long array for the given metadata key. Value type: {@link MetadataValueType#ARRAY}
     * Component type: {@link MetadataValueType#INT64}
     *
     * @param key the metadata key
     * @param value the long array
     * @return this builder instance
     */
    Builder putArrayOfLong(String key, long[] value);

    /**
     * Sets an unsigned long array for the given metadata key. Value type: {@link
     * MetadataValueType#ARRAY} Component type: {@link MetadataValueType#UINT64}
     *
     * @param key the metadata key
     * @param value the long array (interpreted as unsigned)
     * @return this builder instance
     */
    Builder putArrayOfUnsignedLong(String key, long[] value);

    /**
     * Sets a float array for the given metadata key. Value type: {@link MetadataValueType#ARRAY}
     * Component type: {@link MetadataValueType#FLOAT32}
     *
     * @param key the metadata key
     * @param value the float array
     * @return this builder instance
     */
    Builder putArrayOfFloat(String key, float[] value);

    /**
     * Sets a double array for the given metadata key. Value type: {@link MetadataValueType#ARRAY}
     * Component type: {@link MetadataValueType#FLOAT64}
     *
     * @param key the metadata key
     * @param value the double array
     * @return this builder instance
     */
    Builder putArrayOfDouble(String key, double[] value);
}
