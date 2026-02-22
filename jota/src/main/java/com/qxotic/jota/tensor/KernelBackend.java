package com.qxotic.jota.tensor;

public interface KernelBackend {
    KernelExecutable compile(KernelProgram program, KernelCacheKey cacheKey);

    KernelExecutable load(KernelProgram program, KernelCacheKey cacheKey);

    KernelExecutable getOrCompile(KernelProgram program, KernelCacheKey cacheKey);

    KernelExecutableCache cache();

    interface KernelExecutableCache {
        KernelExecutable get(KernelCacheKey key);

        void put(KernelCacheKey key, KernelExecutable exec);
    }
}
