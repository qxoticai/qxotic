package ai.qxotic.jota.c;

import ai.qxotic.jota.ir.lir.LIRGraph;
import ai.qxotic.jota.ir.lir.scratch.ScratchLayout;
import ai.qxotic.jota.tensor.KernelBackend;
import ai.qxotic.jota.tensor.KernelCacheKey;
import ai.qxotic.jota.tensor.KernelExecutable;
import ai.qxotic.jota.tensor.KernelProgram;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

final class CKernelBackend implements KernelBackend {

    private final KernelExecutableCache cache = new InMemoryKernelCache();
    private final CKernelCompiler compiler = new CKernelCompiler();

    @Override
    public KernelExecutable compile(KernelProgram program, KernelCacheKey cacheKey) {
        CNative.requireAvailable();
        CKernelSpec spec = compiler.compile(program, cacheKey);
        long fnPtr = CNative.loadKernel(spec.soPath().toString(), spec.kernelName());
        return new CKernelExecutable(fnPtr);
    }

    @Override
    public KernelExecutable load(KernelProgram program, KernelCacheKey cacheKey) {
        throw new UnsupportedOperationException("C backend does not load binary kernels");
    }

    @Override
    public KernelExecutable getOrCompile(KernelProgram program, KernelCacheKey cacheKey) {
        KernelExecutable exec = cache.get(cacheKey);
        if (exec != null) {
            return exec;
        }
        KernelExecutable created = compile(program, cacheKey);
        cache.put(cacheKey, created);
        return created;
    }

    KernelCacheKey cacheKey(LIRGraph graph, ScratchLayout scratchLayout) {
        return compiler.cacheKey(graph, scratchLayout);
    }

    @Override
    public KernelExecutableCache cache() {
        return cache;
    }

    private static final class InMemoryKernelCache implements KernelExecutableCache {
        private final Map<KernelCacheKey, KernelExecutable> map = new ConcurrentHashMap<>();

        @Override
        public KernelExecutable get(KernelCacheKey key) {
            return map.get(key);
        }

        @Override
        public void put(KernelCacheKey key, KernelExecutable exec) {
            map.put(key, exec);
        }
    }
}
