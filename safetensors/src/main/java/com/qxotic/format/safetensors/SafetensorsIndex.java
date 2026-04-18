package com.qxotic.format.safetensors;

import com.qxotic.format.safetensors.impl.ImplAccessor;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Collection;
import java.util.Objects;
import java.util.Optional;

/** Index for locating tensors across one or more safetensors files. */
public interface SafetensorsIndex {
    Path getRootPath();

    /** Returns the safetensors file path containing the tensor, or null if not found. */
    Path getSafetensorsPath(String tensorName);

    default boolean containsTensor(String tensorName) {
        return getSafetensorsPath(tensorName) != null;
    }

    /** Returns the safetensors path containing the tensor as an Optional. */
    default Optional<Path> findSafetensorsPath(String tensorName) {
        Objects.requireNonNull(tensorName, "tensorName");
        return Optional.ofNullable(getSafetensorsPath(tensorName));
    }

    /** Returns the safetensors path containing the tensor, or throws if absent. */
    default Path requireSafetensorsPath(String tensorName) {
        Objects.requireNonNull(tensorName, "tensorName");
        return findSafetensorsPath(tensorName)
                .orElseThrow(() -> new IllegalArgumentException("tensor not found: " + tensorName));
    }

    /** All tensor names (order preserved). */
    Collection<String> getTensorNames();

    /**
     * Loads a Safetensors index from a directory or single file. For single-file models, scans
     * tensor names. For sharded models, reads index.json.
     */
    static SafetensorsIndex load(Path rootPath) throws IOException {
        Objects.requireNonNull(rootPath, "rootPath");
        return ImplAccessor.loadIndex(rootPath);
    }
}
