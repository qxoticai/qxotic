package com.qxotic.jota.runtime;

import java.nio.file.Path;

public interface KernelCache {

    Path root();

    KernelCacheEntry entryFor(KernelCacheKey key);
}
