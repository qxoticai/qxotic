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
     * @throws NullPointerException if gguf is null
     */
    static Builder newBuilder(GGUF gguf) {
        return ImplAccessor.newBuilder(gguf);
    }

    /** Creates a new empty {@link Builder}. */
    static Builder newBuilder() {
        return ImplAccessor.newBuilder();
    }

    /**
     * Builds a GGUF instance with automatic tensor offset computation.
     *
     * <p>This is equivalent to calling {@code build(true)}, which will recompute all tensor offsets
     * to ensure proper alignment and packing.
     *
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
     */
    GGUF build(boolean recomputeTensorOffsets);

    /** Creates and returns a copy of this builder. */
    Builder clone();

    /**
     * Sets the GGUF format version.
     *
     * <p>The format version determines how the GGUF file is structured. Version 3 is the current
     * version and supports all features including large file support and all GGML types.
     */
    Builder setVersion(int newVersion);

    /** Gets the GGUF format version. */
    int getVersion();

    /**
     * Sets the alignment value for tensor data.
     *
     * <p>Alignment ensures that tensor data starts at memory addresses that are multiples of this
     * value. This is important for performance on many architectures. Common values are 32 or 64.
     *
     * <p>The alignment must be a power of 2 (e.g., 1, 2, 4, 8, 16, 32, 64).
     *
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
     * @throws NullPointerException if tensorEntry is null
     * @see TensorEntry#create(String, long[], GGMLType, long)
     */
    Builder putTensor(TensorEntry tensorEntry);

    /**
     * Removes a tensor by name.
     *
     * <p>If no tensor with the given name exists, this method does nothing.
     */
    Builder removeTensor(String tensorName);

    /** Checks if a tensor exists by name. */
    boolean containsTensor(String tensorName);

    /** Gets tensor information by name. */
    TensorEntry getTensor(String tensorName);

    /** Checks if a metadata key exists. */
    boolean containsKey(String key);

    /**
     * Gets a metadata value associated with the given key, casting it to the specified target
     * class, or null if the key is not found.
     *
     * <p>See {@link GGUF#getValue(Class, String)} for details on type mapping and examples.
     *
     * @see GGUF#getValue(Class, String)
     */
    <T> T getValue(Class<T> targetClass, String key);

    /**
     * Gets all metadata keys.
     *
     * <p>The iteration order of the returned set preserves the insertion order of keys.
     */
    Set<String> getMetadataKeys();

    /**
     * Gets all tensors.
     *
     * <p>The iteration order of the returned collection preserves the insertion order of tensors.
     */
    Collection<TensorEntry> getTensors();

    /**
     * Gets the component type for the array value associated with the given key.
     *
     * <p>This method returns the element type of arrays stored in metadata. For example, if the key
     * maps to a {@code float[]}, this returns {@link MetadataValueType#FLOAT32}.
     *
     * @see MetadataValueType#ARRAY
     */
    MetadataValueType getComponentType(String key);

    /** Gets the type of the metadata value associated with the given key. */
    MetadataValueType getType(String key);

    /**
     * Removes a metadata key.
     *
     * <p>If the key does not exist, this method does nothing.
     */
    Builder removeKey(String key);

    /**
     * Sets a String for the given metadata key. Value type: {@link MetadataValueType#STRING}
     *
     * @return this builder instance
     */
    Builder putString(String key, String value);

    /**
     * Sets a boolean for the given metadata key. Value type: {@link MetadataValueType#BOOL}
     *
     * @return this builder instance
     */
    Builder putBoolean(String key, boolean value);

    /**
     * Sets a byte for the given metadata key. Value type: {@link MetadataValueType#INT8}
     *
     * @return this builder instance
     */
    Builder putByte(String key, byte value);

    /**
     * Sets an unsigned byte for the given metadata key. Value type: {@link MetadataValueType#UINT8}
     *
     * @return this builder instance
     */
    Builder putUnsignedByte(String key, byte value);

    /**
     * Sets a short for the given metadata key. Value type: {@link MetadataValueType#INT16}
     *
     * @return this builder instance
     */
    Builder putShort(String key, short value);

    /**
     * Sets an unsigned short for the given metadata key. Value type: {@link
     * MetadataValueType#UINT16}
     *
     * @return this builder instance
     */
    Builder putUnsignedShort(String key, short value);

    /**
     * Sets an integer for the given metadata key. Value type: {@link MetadataValueType#INT32}
     *
     * @return this builder instance
     */
    Builder putInteger(String key, int value);

    /**
     * Sets an unsigned integer for the given metadata key. Value type: {@link
     * MetadataValueType#UINT32}
     *
     * @return this builder instance
     */
    Builder putUnsignedInteger(String key, int value);

    /**
     * Sets a long for the given metadata key. Value type: {@link MetadataValueType#INT64}
     *
     * @return this builder instance
     */
    Builder putLong(String key, long value);

    /**
     * Sets an unsigned long for the given metadata key. Value type: {@link
     * MetadataValueType#UINT64}
     *
     * @return this builder instance
     */
    Builder putUnsignedLong(String key, long value);

    /**
     * Sets a float for the given metadata key. Value type: {@link MetadataValueType#FLOAT32}
     *
     * @return this builder instance
     */
    Builder putFloat(String key, float value);

    /**
     * Sets a double for the given metadata key. Value type: {@link MetadataValueType#FLOAT64}
     *
     * @return this builder instance
     */
    Builder putDouble(String key, double value);

    /**
     * Sets a boolean array for the given metadata key. Value type: {@link MetadataValueType#ARRAY}
     * Component type: {@link MetadataValueType#BOOL}
     *
     * @return this builder instance
     */
    Builder putArrayOfBoolean(String key, boolean[] value);

    /**
     * Sets a String array for the given metadata key. Value type: {@link MetadataValueType#ARRAY}
     * Component type: {@link MetadataValueType#STRING}
     *
     * @return this builder instance
     */
    Builder putArrayOfString(String key, String[] value);

    /**
     * Sets a byte array for the given metadata key. Value type: {@link MetadataValueType#ARRAY}
     * Component type: {@link MetadataValueType#INT8}
     *
     * @return this builder instance
     */
    Builder putArrayOfByte(String key, byte[] value);

    /**
     * Sets an unsigned byte array for the given metadata key. Value type: {@link
     * MetadataValueType#ARRAY} Component type: {@link MetadataValueType#UINT8}
     *
     * @return this builder instance
     */
    Builder putArrayOfUnsignedByte(String key, byte[] value);

    /**
     * Sets a short array for the given metadata key. Value type: {@link MetadataValueType#ARRAY}
     * Component type: {@link MetadataValueType#INT16}
     *
     * @return this builder instance
     */
    Builder putArrayOfShort(String key, short[] value);

    /**
     * Sets an unsigned short array for the given metadata key. Value type: {@link
     * MetadataValueType#ARRAY} Component type: {@link MetadataValueType#UINT16}
     *
     * @return this builder instance
     */
    Builder putArrayOfUnsignedShort(String key, short[] value);

    /**
     * Sets an integer array for the given metadata key. Value type: {@link MetadataValueType#ARRAY}
     * Component type: {@link MetadataValueType#INT32}
     *
     * @return this builder instance
     */
    Builder putArrayOfInteger(String key, int[] value);

    /**
     * Sets an unsigned integer array for the given metadata key. Value type: {@link
     * MetadataValueType#ARRAY} Component type: {@link MetadataValueType#UINT32}
     *
     * @return this builder instance
     */
    Builder putArrayOfUnsignedInteger(String key, int[] value);

    /**
     * Sets a long array for the given metadata key. Value type: {@link MetadataValueType#ARRAY}
     * Component type: {@link MetadataValueType#INT64}
     *
     * @return this builder instance
     */
    Builder putArrayOfLong(String key, long[] value);

    /**
     * Sets an unsigned long array for the given metadata key. Value type: {@link
     * MetadataValueType#ARRAY} Component type: {@link MetadataValueType#UINT64}
     *
     * @return this builder instance
     */
    Builder putArrayOfUnsignedLong(String key, long[] value);

    /**
     * Sets a float array for the given metadata key. Value type: {@link MetadataValueType#ARRAY}
     * Component type: {@link MetadataValueType#FLOAT32}
     *
     * @return this builder instance
     */
    Builder putArrayOfFloat(String key, float[] value);

    /**
     * Sets a double array for the given metadata key. Value type: {@link MetadataValueType#ARRAY}
     * Component type: {@link MetadataValueType#FLOAT64}
     *
     * @return this builder instance
     */
    Builder putArrayOfDouble(String key, double[] value);
}
