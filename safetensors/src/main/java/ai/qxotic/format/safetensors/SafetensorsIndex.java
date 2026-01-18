package ai.qxotic.format.safetensors;

import ai.qxotic.format.safetensors.impl.ImplAccessor;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Collection;

/**
 * Index for locating tensors across one or more safetensors files.
 *
 * <p>Handles both single-file and sharded models transparently. For sharded models, uses lazy
 * loading to parse files only when accessed.
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
     * Loads a safetensors model directory.
     *
     * <p>For single-file models, parses the file immediately. For sharded models, only parses the
     * index.json and loads individual files lazily on access.
     *
     * @param rootPath the model root directory
     * @return the index
     * @throws IOException if I/O error or missing required files
     */
    static SafetensorsIndex load(Path rootPath) throws IOException {
        return ImplAccessor.loadIndex(rootPath);
    }
}
