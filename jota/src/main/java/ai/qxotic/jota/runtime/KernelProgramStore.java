package ai.qxotic.jota.runtime;

import ai.qxotic.jota.tensor.KernelCacheKey;
import ai.qxotic.jota.tensor.KernelProgram;
import java.nio.file.Path;
import java.util.Optional;

public interface KernelProgramStore {
    Path root();

    void store(KernelProgram program, KernelCacheKey key);

    Optional<KernelProgram> load(KernelCacheKey key);
}
