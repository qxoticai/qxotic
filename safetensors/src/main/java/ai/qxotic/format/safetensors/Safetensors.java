package ai.qxotic.format.safetensors;

import ai.qxotic.format.safetensors.impl.ImplAccessor;
import java.io.IOException;
import java.nio.channels.ReadableByteChannel;
import java.nio.channels.WritableByteChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Collection;
import java.util.Map;

/**
 * Interface for reading and writing Safetensors headers (metadata + tensor entries).
 *
 * <p>This API works on headers only. It reads and writes metadata plus tensor descriptors ({@link
 * TensorEntry}) and does not read or write tensor payload bytes.
 *
 * <p>Format specification:
 *
 * <ul>
 *   <li>8 bytes: header size (N) as unsigned little-endian 64-bit integer
 *   <li>N bytes: JSON UTF-8 header (must start with '{', may have trailing 0x20 padding)
 *   <li>Rest: tensor payload bytes
 * </ul>
 *
 * <p>Duplicate JSON keys in headers follow the underlying JSON parser policy (last key wins).
 *
 * @see <a href="https://github.com/huggingface/safetensors">Safetensors specification</a>
 */
public interface Safetensors {

    /**
     * Returns the byte offset where tensor payload begins. This is always {@code 8 + headerSize}.
     *
     * @return byte offset to start of tensor payload bytes
     */
    long getTensorDataOffset();

    /**
     * Returns tensor alignment in bytes.
     *
     * <p>If {@code __alignment__} is not present in metadata, this is the default alignment.
     *
     * @return alignment in bytes
     */
    int getAlignment();

    /**
     * Returns metadata from the __metadata__ key. Per spec, all values must be strings.
     *
     * @return unmodifiable map of metadata, empty if no __metadata__ present
     */
    Map<String, String> getMetadata();

    /**
     * Returns all tensors in this file, order is preserved.
     *
     * @return unmodifiable collection of tensor information
     */
    Collection<TensorEntry> getTensors();

    /**
     * Returns information for a specific tensor.
     *
     * @param tensorName the tensor name
     * @return tensor information, or null if not found
     */
    TensorEntry getTensor(String tensorName);

    /**
     * Checks if a tensor exists.
     *
     * @param tensorName the tensor name
     * @return true if tensor exists
     */
    default boolean containsTensor(String tensorName) {
        return getTensor(tensorName) != null;
    }

    /**
     * Reads safetensors metadata from a channel. Only reads the header (8 bytes + N bytes JSON),
     * not tensor payload data.
     *
     * @param channel the channel to read from
     * @return safetensors metadata
     * @throws IOException if I/O error or format violation
     */
    static Safetensors read(ReadableByteChannel channel) throws IOException {
        return ImplAccessor.read(channel);
    }

    /**
     * Reads safetensors metadata from a file.
     *
     * @param path the file path
     * @return safetensors metadata
     * @throws IOException if I/O error or format violation
     */
    static Safetensors read(Path path) throws IOException {
        try (ReadableByteChannel channel = Files.newByteChannel(path, StandardOpenOption.READ)) {
            return read(channel);
        }
    }

    /**
     * Writes a Safetensors header to a {@link WritableByteChannel}.
     *
     * @param safetensors the Safetensors instance to write
     * @param byteChannel the channel to write to
     * @throws IOException if an I/O error occurs during writing
     */
    static void write(Safetensors safetensors, WritableByteChannel byteChannel) throws IOException {
        ImplAccessor.write(safetensors, byteChannel);
    }

    /**
     * Writes a Safetensors header to a file at the specified path.
     *
     * @param safetensors the Safetensors instance to write
     * @param modelPath the path where the Safetensors file should be written
     * @throws IOException if an I/O error occurs during writing
     */
    static void write(Safetensors safetensors, Path modelPath) throws IOException {
        try (WritableByteChannel byteChannel =
                Files.newByteChannel(
                        modelPath, StandardOpenOption.WRITE, StandardOpenOption.CREATE_NEW)) {
            write(safetensors, byteChannel);
        }
    }
}
