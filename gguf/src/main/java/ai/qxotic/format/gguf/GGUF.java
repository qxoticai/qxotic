package ai.qxotic.format.gguf;

import ai.qxotic.format.gguf.impl.ImplAccessor;
import java.io.IOException;
import java.nio.channels.ReadableByteChannel;
import java.nio.channels.WritableByteChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Collection;
import java.util.Set;

/**
 * Interface for handling GGUF files, which are used to store
 * large language models and their associated metadata.
 * <p>
 * This interface provides methods to read and write GGUF files, access their metadata,
 * and manage tensor information. GGUF is a binary format that includes both model weights
 * and associated configuration data.
 *
 * @see <a href="https://github.com/ggerganov/ggml/blob/master/docs/gguf.md">GGUF format specification</a>
 */
public interface GGUF {
    /**
     * Returns the version number of the GGUF instance.
     *
     * @return the GGUF format version as an integer
     */
    int getVersion();

    /**
     * Returns the alignment value used in the GGUF file. If not explicitly set,
     * returns the default alignment value.
     * <p>
     * The alignment determines the byte alignment requirements for tensor data.
     *
     * @return the alignment value, or the default alignment if not specified
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
     * @return the byte offset to the start of tensor data
     */
    long getTensorDataOffset();

    /**
     * Returns a set of all metadata keys present in the GGUF metadata, order is preserved.
     *
     * @return a set containing all metadata keys
     */
    Set<String> getMetadataKeys();

    /**
     * Gets a metadata value associated with the given key, casting it to the specified target class,
     * or null if the key is not found.
     * The method handles primitive types, wrapper classes, strings, and arrays.
     *
     * <p>The actual type of the stored value depends on {@link #getType(String)}:
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
     *     <ul>
     *       <li>{@code STRING} → {@code String[]}
     *       <li>{@code UINT8} → {@code byte[]} (signed values)
     *       <li>{@code INT8} → {@code byte[]}
     *       <li>{@code UINT16} → {@code short[]} (signed values)
     *       <li>{@code INT16} → {@code short[]}
     *       <li>{@code UINT32} → {@code int[]} (signed values)
     *       <li>{@code INT32} → {@code int[]}
     *       <li>{@code UINT64} → {@code long[]} (signed values)
     *       <li>{@code INT64} → {@code long[]}
     *       <li>{@code FLOAT32} → {@code float[]}
     *       <li>{@code FLOAT64} → {@code double[]}
     *       <li>{@code BOOL} → {@code boolean[]}
     *     </ul>
     * </ul>
     *
     * <p>Examples:
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
     * @param <T>         the type to cast the value to
     * @param targetClass the Class object representing the desired return type
     * @param key         the key whose associated value is to be returned
     * @return the value associated with the key, cast to type T, or null if the key is not found
     * @throws ClassCastException if the value cannot be cast to the requested type or if the
     *                            requested type doesn't match the type indicated by {@link #getType(String)}
     * @see #getType(String)
     * @see #getComponentType(String)
     */
    <T> T getValue(Class<T> targetClass, String key);

    /**
     * Retrieves the value associated with the specified metadata key, or returns
     * a default value if the key is not present.
     *
     * @param <T>          the expected type of the value
     * @param key          the metadata key to look up
     * @param defaultValue the value to return if the key is not found
     * @return the value associated with the key, or defaultValue if not found
     * @see #getValue(Class, String)
     */
    default <T> T getValueOrDefault(Class<T> targetClass, String key, T defaultValue) {
        return containsKey(key) ? getValue(targetClass, key) : defaultValue;
    }

    /**
     * Checks if a metadata key exists.
     *
     * @param key the metadata key to check
     * @return true if the key exists, false otherwise
     */
    default boolean containsKey(String key) {
        return getValue(Object.class, key) != null;
    }

    /**
     * Returns the metadata value type for the specified key.
     *
     * @param key the metadata key to look up
     * @return the {@link MetadataValueType} of the value associated with the key, or null if not found
     */
    MetadataValueType getType(String key);

    /**
     * Returns the component type for {@link MetadataValueType#ARRAY array} metadata values.
     *
     * @param key the metadata key to look up
     * @return the {@link MetadataValueType} of the array components, or null if not found or the value associated with the given key is not an {@link MetadataValueType#ARRAY array}
     */
    MetadataValueType getComponentType(String key);

    /**
     * Returns information about all tensors stored in the GGUF metadata, order is preserved.
     *
     * @return a collection of {@link TensorInfo} objects describing all tensors
     */
    Collection<TensorInfo> getTensors();

    /**
     * Returns information about a specific tensor by name.
     *
     * @param tensorName the name of the tensor to look up
     * @return the {@link TensorInfo} for the specified tensor, or null if not found
     */
    TensorInfo getTensor(String tensorName);

    /**
     * Checks if a tensor with the specified name exists in the GGUF file.
     *
     * @param tensorName the name of the tensor to check
     * @return true if the tensor exists, false otherwise
     */
    default boolean containsTensor(String tensorName) {
        return getTensor(tensorName) != null;
    }

    /**
     * Reads GGUF metadata from a {@link ReadableByteChannel}.
     *
     * @param byteChannel the channel to read from
     * @return a new GGUF instance containing the metadata
     * @throws IOException if an I/O error occurs during reading
     */
    static GGUF read(ReadableByteChannel byteChannel) throws IOException {
        return ImplAccessor.read(byteChannel);
    }

    /**
     * Reads a GGUF file from a path.
     *
     * @param modelPath the path to the GGUF file
     * @return a new GGUF instance containing the medatada
     * @throws IOException if an I/O error occurs during reading
     */
    static GGUF read(Path modelPath) throws IOException {
        try (ReadableByteChannel byteChannel = Files.newByteChannel(modelPath, StandardOpenOption.READ)) {
            return read(byteChannel);
        }
    }

    /**
     * Writes GGUF metadata to a {@link WritableByteChannel}.
     *
     * @param gguf        the GGUF instance to write
     * @param byteChannel the channel to write to
     * @throws IOException if an I/O error occurs during writing
     */
    static void write(GGUF gguf, WritableByteChannel byteChannel) throws IOException {
        ImplAccessor.write(gguf, byteChannel);
    }

    /**
     * Writes a GGUF instance to a file at the specified path.
     *
     * @param gguf      the GGUF instance to write
     * @param modelPath the path where the GGUF file should be written
     * @throws IOException if an I/O error occurs during writing
     */
    static void write(GGUF gguf, Path modelPath) throws IOException {
        try (WritableByteChannel byteChannel =
                Files.newByteChannel(modelPath, StandardOpenOption.WRITE, StandardOpenOption.CREATE_NEW)) {
            write(gguf, byteChannel);
        }
    }
}
