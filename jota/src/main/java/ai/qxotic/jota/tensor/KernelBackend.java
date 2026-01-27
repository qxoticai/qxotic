package ai.qxotic.jota.tensor;

public interface KernelBackend {
    KernelExecutable compile(KernelProgram program, KernelCacheKey cacheKey);

    KernelExecutable load(KernelProgram program, KernelCacheKey cacheKey);

    KernelExecutable getOrCompile(KernelProgram program, KernelCacheKey cacheKey);

    LaunchConfig chooseLaunch(ExpressionGraph graph, LaunchHints hints);

    KernelCacheKey cacheKey(ExpressionGraph graph);

    KernelExecutableCache cache();

    interface KernelExecutableCache {
        KernelExecutable get(KernelCacheKey key);

        void put(KernelCacheKey key, KernelExecutable exec);
    }
}
