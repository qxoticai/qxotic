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
 * Read and write access to GGUF files, the binary format used by GGML-based models to store
 * metadata and tensor descriptors. Only the metadata section is read or written; tensor payload
 * bytes are not accessed.
 *
 * <p>The header structure is read-only, but array-valued metadata is returned by reference. Do not
 * mutate those arrays while an instance is shared between threads.
 *
 * @see <a href="https://github.com/ggml-org/ggml/blob/master/docs/gguf.md">GGUF format
 *     specification</a>
 */
public interface GGUF {
    /** GGUF format version. */
    int getVersion();

    /**
     * Tensor data alignment in bytes; returns the default when {@code general.alignment} is absent.
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

    /** Absolute byte offset where tensor data begins in the file. */
    long getTensorDataOffset();

    /** All metadata keys, insertion order preserved. */
    Set<String> getMetadataKeys();

    /**
     * Metadata value for {@code key} cast to {@code targetClass}, or {@code null} if the key is
     * absent. Unsigned types map to the signed Java type of the same width (e.g. {@code UINT32} to
     * {@code int}); unsigned conversion is up to the caller. Pass {@code Object.class} for
     * unchecked access.
     *
     * <p>Array values are returned by reference, not copied.
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
     * @throws ClassCastException if the value cannot be cast to the requested type or if the
     *     requested type doesn't match the type indicated by {@link #getType(String)}
     * @see #getComponentType(String)
     */
    <T> T getValue(Class<T> targetClass, String key);

    /** Metadata value for {@code key}, or {@code defaultValue} if the key is absent. */
    default <T> T getValueOrDefault(Class<T> targetClass, String key, T defaultValue) {
        return containsKey(key) ? getValue(targetClass, key) : defaultValue;
    }

    /**
     * Equivalent to {@code getValue(String.class, key)}.
     *
     * @throws ClassCastException if the value is not a string
     */
    default String getString(String key) {
        return getValue(String.class, key);
    }

    /**
     * Equivalent to {@code getValueOrDefault(String.class, key, defaultValue)}.
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

    /** Type of the metadata value for the key. */
    MetadataValueType getType(String key);

    /** Component type for {@link MetadataValueType#ARRAY array} metadata values. */
    MetadataValueType getComponentType(String key);

    /** All tensors, order preserved. */
    Collection<TensorEntry> getTensors();

    /** Tensor entry for the name, or {@code null} if absent. */
    TensorEntry getTensor(String tensorName);

    /** Checks if a tensor with the specified name exists in the GGUF file. */
    default boolean containsTensor(String tensorName) {
        return getTensor(tensorName) != null;
    }

    /**
     * Absolute byte offset of the tensor's data: {@code getTensorDataOffset() + tensor.offset()}.
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

    /** Detailed string representation with control over what to display. */
    default String toString(boolean showKeys, boolean showTensors) {
        return ImplAccessor.toString(this, showKeys, showTensors);
    }

    /**
     * Detailed string representation with control over what to display and how long arrays and
     * strings may be before they are elided.
     */
    default String toString(
            boolean showKeys, boolean showTensors, int maxArrayElements, int maxStringLength) {
        return ImplAccessor.toString(
                this, showKeys, showTensors, maxArrayElements, maxStringLength);
    }
}
