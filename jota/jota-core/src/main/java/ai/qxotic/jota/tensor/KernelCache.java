package ai.qxotic.jota.tensor;

import java.nio.file.Path;

public interface KernelCache {

    Path root();

    KernelCacheEntry entryFor(KernelCacheKey key);
}
