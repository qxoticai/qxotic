package com.qxotic.jota.runtime;

import java.nio.file.Path;
import java.util.Optional;

public interface KernelProgramStore {
    Path root();

    void store(KernelProgram program, KernelCacheKey key);

    Optional<KernelProgram> load(KernelCacheKey key);
}
