package com.qxotic.jota.runtime;

import com.qxotic.jota.tensor.KernelCacheKey;
import com.qxotic.jota.tensor.KernelProgram;
import java.nio.file.Path;
import java.util.Optional;

public interface KernelProgramStore {
    Path root();

    void store(KernelProgram program, KernelCacheKey key);

    Optional<KernelProgram> load(KernelCacheKey key);
}
