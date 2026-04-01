package com.qxotic.format.gguf;

import com.qxotic.format.gguf.impl.ImplAccessor;
import java.io.BufferedInputStream;
import java.io.IOException;
import java.nio.channels.Channels;
import java.nio.channels.ReadableByteChannel;
import java.nio.channels.WritableByteChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Collection;
import java.util.Objects;
import java.util.Set;

/**
 * Interface for handling GGUF files, which are used to store large language models and their
 * associated metadata.
 *
 * <p>This interface provides methods to read and write GGUF files, access their metadata, and
 * manage tensor information. GGUF is a binary format that includes both model weights and
 * associated configuration data.
 *
 * @see <a href="https://github.com/ggml-org/ggml/blob/master/docs/gguf.md">GGUF format
 *     specification</a>
 */
public interface GGUF {
    /**
     * Returns the version number of the GGUF instance.
     *
     * @return the GGUF format version
     */
    int getVersion();

    /**
     * Returns the alignment value used in the GGUF file. If not explicitly set, returns the default
     * alignment value.
     *
     * <p>The alignment determines the byte alignment requirements for tensor data.
     *
     * @return the alignment value in bytes
     */
    default int getAlignment() {
        if (containsKey(ImplAccessor.alignmentKey())) {
            assert getType(ImplAccessor.alignmentKey()) == MetadataValueType.UINT32;
            return getValue(int.class, ImplAccessor.alignmentKey());
        }
        return ImplAccessor.defaultAlignment();
    }

    /**
     * Returns the byte offset where tensor data begins in the GGUF file.
     *
     * @return the tensor data offset in bytes
     */
    long getTensorDataOffset();

    /**
     * Returns a set of all metadata keys present in the GGUF metadata, order is preserved.
     *
     * @return the metadata keys
     */
    Set<String> getMetadataKeys();

    /**
     * Gets a metadata value associated with the given key, casting it to the specified target
     * class, or null if the key is not found. The method handles primitive types, wrapper classes,
     * strings, and arrays.
     *
     * <p>The actual type of the stored value depends on {@link #getType(String)}:
     *
     * <ul>
     *   <li>{@code UINT8} → {@code byte} (signed, may require manual unsigned conversion)
     *   <li>{@code INT8} → {@code byte}
     *   <li>{@code UINT16} → {@code short} (signed, may require manual unsigned conversion)
     *   <li>{@code INT16} → {@code short}
     *   <li>{@code UINT32} → {@code int} (signed, may require manual unsigned conversion)
     *   <li>{@code INT32} → {@code int}
     *   <li>{@code FLOAT32} → {@code float}
     *   <li>{@code BOOL} → {@code boolean}
     *   <li>{@code STRING} → {@code String}
     *   <li>{@code UINT64} → {@code long} (signed, may require manual unsigned conversion)
     *   <li>{@code INT64} → {@code long}
     *   <li>{@code FLOAT64} → {@code double}
     *   <li>{@code ARRAY} → Array type depends on {@link #getComponentType(String)}:
     *       <ul>
     *         <li>{@code STRING} → {@code String[]}
     *         <li>{@code UINT8} → {@code byte[]} (signed values)
     *         <li>{@code INT8} → {@code byte[]}
     *         <li>{@code UINT16} → {@code short[]} (signed values)
     *         <li>{@code INT16} → {@code short[]}
     *         <li>{@code UINT32} → {@code int[]} (signed values)
     *         <li>{@code INT32} → {@code int[]}
     *         <li>{@code UINT64} → {@code long[]} (signed values)
     *         <li>{@code INT64} → {@code long[]}
     *         <li>{@code FLOAT32} → {@code float[]}
     *         <li>{@code FLOAT64} → {@code double[]}
     *         <li>{@code BOOL} → {@code boolean[]}
     *       </ul>
     * </ul>
     *
     * <p>Examples:
     *
     * <pre>{@code
     * // Primitive types
     * // Will throw NullPointerException is the key is not present.
     * int intValue = getValue(int.class, "numberKey");
     * boolean flag = getValue(boolean.class, "flagKey");
     *
     * // Wrapper classes
     * Integer boxedInt = getValue(Integer.class, "numberKey");
     * Boolean boxedFlag = getValue(Boolean.class, "flagKey");
     *
     * // Strings
     * String text = getValue(String.class, "textKey");
     *
     * // Arrays
     * int[] numbers = getValue(int[].class, "numberArrayKey");
     * float[] floats = getValue(float[].class, "floatArrayKey");
     * String[] strings = getValue(String[].class, "stringArrayKey");
     *
     * // For generic access without type checking
     * Object generic = getValue(Object.class, "anyKey");
     * }</pre>
     *
     * @throws ClassCastException if the value cannot be cast to the requested type or if the
     *     requested type doesn't match the type indicated by {@link #getType(String)}
     * @see #getType(String)
     * @see #getComponentType(String)
     */
    <T> T getValue(Class<T> targetClass, String key);

    /**
     * Retrieves the value associated with the specified metadata key, or returns a default value if
     * the key is not present.
     *
     * @see #getValue(Class, String)
     */
    default <T> T getValueOrDefault(Class<T> targetClass, String key, T defaultValue) {
        return containsKey(key) ? getValue(targetClass, key) : defaultValue;
    }

    /**
     * Gets a string value for the given key.
     *
     * <p>This is a convenience method equivalent to {@code getValue(String.class, key)}.
     *
     * @throws ClassCastException if the value is not a string
     */
    default String getString(String key) {
        return getValue(String.class, key);
    }

    /**
     * Gets a string value for the given key, or returns the default value if the key is not found.
     *
     * <p>This is a convenience method equivalent to {@code getValueOrDefault(String.class, key,
     * defaultValue)}.
     *
     * @throws ClassCastException if the value is not a string
     */
    default String getStringOrDefault(String key, String defaultValue) {
        return getValueOrDefault(String.class, key, defaultValue);
    }

    /** Checks if a metadata key exists. */
    default boolean containsKey(String key) {
        return getValue(Object.class, key) != null;
    }

    /** Returns the metadata value type for the specified key. */
    MetadataValueType getType(String key);

    /** Returns the component type for {@link MetadataValueType#ARRAY array} metadata values. */
    MetadataValueType getComponentType(String key);

    /** Returns information about all tensors stored in the GGUF metadata, order is preserved. */
    Collection<TensorEntry> getTensors();

    /** Returns information about a specific tensor by name. */
    TensorEntry getTensor(String tensorName);

    /** Checks if a tensor with the specified name exists in the GGUF file. */
    default boolean containsTensor(String tensorName) {
        return getTensor(tensorName) != null;
    }

    /**
     * Returns the absolute byte offset where the tensor's data begins in the GGUF file.
     *
     * <p>This is a convenience method equivalent to {@code getTensorDataOffset() +
     * tensor.offset()}.
     *
     * <p>Example usage when reading tensor data:
     *
     * <pre>{@code
     * GGUF gguf = GGUF.read(path);
     * TensorEntry tensor = gguf.getTensor("weights");
     * long absoluteOffset = gguf.absoluteOffset(tensor);
     * // Use absoluteOffset to read from file channel
     * }</pre>
     *
     * @throws NullPointerException if tensor is null
     */
    default long absoluteOffset(TensorEntry tensor) {
        Objects.requireNonNull(tensor, "tensor");
        return getTensorDataOffset() + tensor.offset();
    }

    /** Reads GGUF metadata from a {@link ReadableByteChannel}. */
    static GGUF read(ReadableByteChannel byteChannel) throws IOException {
        return ImplAccessor.read(byteChannel);
    }

    /** Reads a GGUF file from a path. */
    static GGUF read(Path modelPath) throws IOException {
        try (ReadableByteChannel byteChannel =
                Channels.newChannel(new BufferedInputStream(Files.newInputStream(modelPath)))) {
            return read(byteChannel);
        }
    }

    /** Writes GGUF metadata to a {@link WritableByteChannel}. */
    static void write(GGUF gguf, WritableByteChannel byteChannel) throws IOException {
        ImplAccessor.write(gguf, byteChannel);
    }

    /** Writes a GGUF instance to a file at the specified path. */
    static void write(GGUF gguf, Path modelPath) throws IOException {
        try (WritableByteChannel byteChannel =
                Files.newByteChannel(
                        modelPath, StandardOpenOption.WRITE, StandardOpenOption.CREATE_NEW)) {
            write(gguf, byteChannel);
        }
    }

    /** Returns a detailed string representation of the GGUF with control over what to display. */
    default String toString(boolean showKeys, boolean showTensors) {
        return ImplAccessor.toString(this, showKeys, showTensors);
    }

    /**
     * Returns a detailed string representation of the GGUF with full control over display and
     * elision.
     */
    default String toString(
            boolean showKeys, boolean showTensors, int maxArrayElements, int maxStringLength) {
        return ImplAccessor.toString(
                this, showKeys, showTensors, maxArrayElements, maxStringLength);
    }
}
