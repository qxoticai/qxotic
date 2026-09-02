package com.qxotic.format.gguf;

import com.qxotic.format.gguf.impl.ImplAccessor;
import java.util.Collection;
import java.util.Set;

/**
 * Fluent builder for creating {@link GGUF} instances or modifying copies of existing ones. Not
 * thread-safe.
 *
 * <p>The builder describes metadata and tensors only; {@link GGUF#write} does not copy or write
 * tensor payload bytes.
 *
 * <p>Array-valued metadata is stored by reference. The builder, its clones, and built {@link GGUF}
 * instances may therefore share those arrays; callers must not mutate them while in use.
 *
 * <p>Creating a new GGUF file:
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
 * <p>Modifying an existing GGUF file:
 *
 * <pre>{@code
 * GGUF modified = Builder.newBuilder(GGUF.read(Path.of("model.gguf")))
 *     .putString("general.description", "Modified model")
 *     .removeKey("deprecated_key")
 *     .build();
 *
 * GGUF.write(modified, Path.of("modified.gguf"));
 * }</pre>
 *
 * @see <a href="https://github.com/ggml-org/ggml/blob/master/docs/gguf.md">GGUF format
 *     specification</a>
 */
public interface Builder extends Cloneable {
    /** Returns a builder pre-populated from {@code gguf}; array metadata remains shared. */
    static Builder newBuilder(GGUF gguf) {
        return ImplAccessor.newBuilder(gguf);
    }

    /** Returns a new empty builder. */
    static Builder newBuilder() {
        return ImplAccessor.newBuilder();
    }

    /**
     * Builds with tensor offsets recomputed for alignment and packing; same as {@code build(true)}.
     */
    default GGUF build() {
        return build(true);
    }

    /**
     * Builds the GGUF instance. When {@code recomputeTensorOffsets} is true, tensors are packed
     * contiguously after the metadata with offsets aligned per {@link #getAlignment()}; when false,
     * existing offsets are preserved, e.g. when editing a file without changing its layout.
     */
    GGUF build(boolean recomputeTensorOffsets);

    /** Returns a structural copy; array-valued metadata remains shared. */
    Builder clone();

    /** Sets the GGUF format version; 3 is the current version. */
    Builder setVersion(int newVersion);

    /** GGUF format version. */
    int getVersion();

    /**
     * Sets the tensor data alignment in bytes; must be a power of 2 (commonly 32 or 64).
     *
     * @throws IllegalArgumentException if alignment is not a positive power of 2
     */
    default Builder setAlignment(int newAlignment) {
        if (newAlignment <= 0 || (newAlignment & (newAlignment - 1)) != 0) {
            throw new IllegalArgumentException(
                    "alignment must be a positive power of two but was " + newAlignment);
        }
        return putUnsignedInteger(ImplAccessor.alignmentKey(), newAlignment);
    }

    /**
     * Current alignment in bytes, or the default when unset.
     *
     * @throws GGUFFormatException if {@code general.alignment} is present but not UINT32
     */
    default int getAlignment() {
        if (containsKey(ImplAccessor.alignmentKey())) {
            if (getType(ImplAccessor.alignmentKey()) != MetadataValueType.UINT32) {
                throw new GGUFFormatException(
                        "general.alignment must be UINT32 but was "
                                + getType(ImplAccessor.alignmentKey()));
            }
            return getValue(int.class, ImplAccessor.alignmentKey());
        }
        return ImplAccessor.defaultAlignment();
    }

    /**
     * Adds or replaces a tensor. The tensor's offset is relative to the tensor data section and is
     * recomputed by {@link #build()} unless disabled.
     */
    Builder putTensor(TensorEntry tensorEntry);

    /** Removes the named tensor; no-op if absent. */
    Builder removeTensor(String tensorName);

    /** Whether a tensor with the given name exists. */
    boolean containsTensor(String tensorName);

    /** Tensor entry for the name, or {@code null} if absent. */
    TensorEntry getTensor(String tensorName);

    /** Whether the metadata key exists. */
    boolean containsKey(String key);

    /**
     * Metadata value for {@code key}, or {@code null} if absent. See {@link GGUF#getValue(Class,
     * String)} for type mapping.
     */
    <T> T getValue(Class<T> targetClass, String key);

    /** All metadata keys (unmodifiable, insertion order preserved). */
    Set<String> getMetadataKeys();

    /** All tensors (unmodifiable, insertion order preserved). */
    Collection<TensorEntry> getTensors();

    /** Element type of an {@link MetadataValueType#ARRAY} value, or {@code null} if absent. */
    MetadataValueType getComponentType(String key);

    /** Type of the metadata value, or {@code null} if absent. */
    MetadataValueType getType(String key);

    /** Removes the metadata key; no-op if absent. */
    Builder removeKey(String key);

    /** Sets a string value ({@link MetadataValueType#STRING}). */
    Builder putString(String key, String value);

    /** Sets a boolean value ({@link MetadataValueType#BOOL}). */
    Builder putBoolean(String key, boolean value);

    /** Sets a byte value ({@link MetadataValueType#INT8}). */
    Builder putByte(String key, byte value);

    /**
     * Sets an unsigned byte value ({@link MetadataValueType#UINT8}), stored as a signed Java byte.
     */
    Builder putUnsignedByte(String key, byte value);

    /** Sets a short value ({@link MetadataValueType#INT16}). */
    Builder putShort(String key, short value);

    /**
     * Sets an unsigned short value ({@link MetadataValueType#UINT16}), stored as a signed Java
     * short.
     */
    Builder putUnsignedShort(String key, short value);

    /** Sets an int value ({@link MetadataValueType#INT32}). */
    Builder putInteger(String key, int value);

    /**
     * Sets an unsigned int value ({@link MetadataValueType#UINT32}), stored as a signed Java int.
     */
    Builder putUnsignedInteger(String key, int value);

    /** Sets a long value ({@link MetadataValueType#INT64}). */
    Builder putLong(String key, long value);

    /**
     * Sets an unsigned long value ({@link MetadataValueType#UINT64}), stored as a signed Java long.
     */
    Builder putUnsignedLong(String key, long value);

    /** Sets a float value ({@link MetadataValueType#FLOAT32}). */
    Builder putFloat(String key, float value);

    /** Sets a double value ({@link MetadataValueType#FLOAT64}). */
    Builder putDouble(String key, double value);

    /** Sets a boolean array ({@link MetadataValueType#ARRAY} of {@link MetadataValueType#BOOL}). */
    Builder putArrayOfBoolean(String key, boolean[] value);

    /**
     * Sets a string array ({@link MetadataValueType#ARRAY} of {@link MetadataValueType#STRING}).
     */
    Builder putArrayOfString(String key, String[] value);

    /** Sets a byte array ({@link MetadataValueType#ARRAY} of {@link MetadataValueType#INT8}). */
    Builder putArrayOfByte(String key, byte[] value);

    /**
     * Sets an unsigned byte array ({@link MetadataValueType#ARRAY} of {@link
     * MetadataValueType#UINT8}).
     */
    Builder putArrayOfUnsignedByte(String key, byte[] value);

    /** Sets a short array ({@link MetadataValueType#ARRAY} of {@link MetadataValueType#INT16}). */
    Builder putArrayOfShort(String key, short[] value);

    /**
     * Sets an unsigned short array ({@link MetadataValueType#ARRAY} of {@link
     * MetadataValueType#UINT16}).
     */
    Builder putArrayOfUnsignedShort(String key, short[] value);

    /** Sets an int array ({@link MetadataValueType#ARRAY} of {@link MetadataValueType#INT32}). */
    Builder putArrayOfInteger(String key, int[] value);

    /**
     * Sets an unsigned int array ({@link MetadataValueType#ARRAY} of {@link
     * MetadataValueType#UINT32}).
     */
    Builder putArrayOfUnsignedInteger(String key, int[] value);

    /** Sets a long array ({@link MetadataValueType#ARRAY} of {@link MetadataValueType#INT64}). */
    Builder putArrayOfLong(String key, long[] value);

    /**
     * Sets an unsigned long array ({@link MetadataValueType#ARRAY} of {@link
     * MetadataValueType#UINT64}).
     */
    Builder putArrayOfUnsignedLong(String key, long[] value);

    /**
     * Sets a float array ({@link MetadataValueType#ARRAY} of {@link MetadataValueType#FLOAT32}).
     */
    Builder putArrayOfFloat(String key, float[] value);

    /**
     * Sets a double array ({@link MetadataValueType#ARRAY} of {@link MetadataValueType#FLOAT64}).
     */
    Builder putArrayOfDouble(String key, double[] value);
}
