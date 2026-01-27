package ai.qxotic.jota.hip;

import ai.qxotic.jota.tensor.ExpressionGraph;
import ai.qxotic.jota.tensor.KernelBackend;
import ai.qxotic.jota.tensor.KernelCacheKey;
import ai.qxotic.jota.tensor.KernelExecutable;
import ai.qxotic.jota.tensor.KernelProgram;
import ai.qxotic.jota.tensor.LaunchConfig;
import ai.qxotic.jota.tensor.LaunchHints;
import ai.qxotic.jota.tensor.GraphHasher;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

final class HipKernelBackend implements KernelBackend {

    private final KernelExecutableCache cache = new InMemoryKernelCache();
    private final HipKernelCompiler compiler = new HipKernelCompiler();

    @Override
    public KernelExecutable compile(KernelProgram program, KernelCacheKey cacheKey) {
        HipKernelSpec spec = compiler.compile(program, cacheKey);
        byte[] hsaco = HipKernelSourceLoader.load(spec.hsacoPath().toString());
        HipModule module = HipModule.load(hsaco);
        HipKernelExecutable exec = new HipKernelExecutable(module.function(spec.kernelName()));
        return new KernelExecutable() {
            @Override
            public void launch(
                    ai.qxotic.jota.tensor.LaunchConfig config,
                    ai.qxotic.jota.tensor.KernelArgs args,
                    ai.qxotic.jota.tensor.ExecutionStream stream) {
                exec.launch(config, args, stream);
            }

            @Override
            public void close() {
                module.close();
            }
        };
    }

    @Override
    public KernelExecutable load(KernelProgram program, KernelCacheKey cacheKey) {
        if (program.kind() != KernelProgram.Kind.BINARY) {
            throw new UnsupportedOperationException("HIP load expects binary program");
        }
        byte[] hsaco = (byte[]) program.payload();
        HipModule module = HipModule.load(hsaco);
        HipKernelExecutable exec = new HipKernelExecutable(module.function(program.entryPoint()));
        return new KernelExecutable() {
            @Override
            public void launch(
                    ai.qxotic.jota.tensor.LaunchConfig config,
                    ai.qxotic.jota.tensor.KernelArgs args,
                    ai.qxotic.jota.tensor.ExecutionStream stream) {
                exec.launch(config, args, stream);
            }

            @Override
            public void close() {
                module.close();
            }
        };
    }

    @Override
    public KernelExecutable getOrCompile(KernelProgram program, KernelCacheKey cacheKey) {
        KernelExecutable exec = cache.get(cacheKey);
        if (exec != null) {
            return exec;
        }
        KernelExecutable created =
                program.kind() == KernelProgram.Kind.BINARY
                        ? load(program, cacheKey)
                        : compile(program, cacheKey);
        cache.put(cacheKey, created);
        return created;
    }

    @Override
    public LaunchConfig chooseLaunch(ExpressionGraph graph, LaunchHints hints) {
        long n = graph.root().layout().shape().size();
        int block = 256;
        int grid = (int) ((n + block - 1) / block);
        return new LaunchConfig(grid, 1, 1, block, 1, 1, 0, false);
    }

    @Override
    public KernelCacheKey cacheKey(ExpressionGraph graph) {
        return GraphHasher.hash(graph);
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
