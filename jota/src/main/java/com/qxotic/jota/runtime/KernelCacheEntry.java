package com.qxotic.jota.runtime;

import java.nio.file.Path;

public record KernelCacheEntry(
        KernelCacheKey key,
        String packageName,
        String className,
        Path directory,
        Path sourcePath,
        Path classOutputDir,
        Path classFilePath) {}
