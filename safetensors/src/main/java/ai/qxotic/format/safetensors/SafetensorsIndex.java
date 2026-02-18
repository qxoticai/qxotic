package ai.qxotic.format.safetensors;

import ai.qxotic.format.safetensors.impl.ImplAccessor;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Collection;
import java.util.Objects;

/**
 * Index for locating tensors across one or more safetensors files.
 *
 * <p>Handles both single-file and sharded models transparently.
 */
public interface SafetensorsIndex {
    /** Returns the model root directory. */
    Path getRootPath();

    /**
     * Returns the safetensors file path containing the tensor.
     *
     * @param tensorName the tensor name
     * @return file path, or null if tensor not found
     */
    Path getSafetensorsPath(String tensorName);

    /**
     * Returns all tensor names in this index.
     *
     * @return collection of tensor names, order is preserved
     */
    Collection<String> getTensorNames();

    /**
     * Loads a Safetensors index from a directory or a single {@code .safetensors} file.
     *
     * <p>For single-file models, scans tensor names from the file. For sharded models, reads the
     * index.json and maps tensor names to shard paths.
     *
     * @param rootPath model directory or single {@code .safetensors} file path
     * @return the index
     * @throws IOException if I/O error or missing required files
     */
    static SafetensorsIndex load(Path rootPath) throws IOException {
        Objects.requireNonNull(rootPath, "rootPath");
        return ImplAccessor.loadIndex(rootPath);
    }
}
