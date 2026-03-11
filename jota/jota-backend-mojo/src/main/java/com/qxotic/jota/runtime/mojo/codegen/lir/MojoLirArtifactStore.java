package com.qxotic.jota.runtime.mojo.codegen.lir;

import com.qxotic.jota.runtime.KernelCacheKey;
import com.qxotic.jota.runtime.KernelProgram;
import com.qxotic.jota.runtime.mojo.MojoCachePaths;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/** Persists Mojo kernel source to cache. */
final class MojoLirArtifactStore {

    private MojoLirArtifactStore() {}

    static void persist(KernelCacheKey key, KernelProgram mojoProgram) {
        Path mojoPath = MojoCachePaths.lirSourcePath(key.value());
        try {
            Files.createDirectories(mojoPath.getParent());
            Files.writeString(mojoPath, (String) mojoProgram.payload());
        } catch (IOException e) {
            throw new IllegalStateException("Failed to persist Mojo LIR artifact", e);
        }
    }
}
