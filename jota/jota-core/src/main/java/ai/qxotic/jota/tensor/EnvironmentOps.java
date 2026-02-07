package ai.qxotic.jota.tensor;

import ai.qxotic.jota.Environment;
import ai.qxotic.jota.ExecutionMode;
import ai.qxotic.jota.memory.MemoryDomain;
import java.util.Collections;
import java.util.Map;
import java.util.WeakHashMap;

final class EnvironmentOps {

    private static final Map<Environment, OpsCache> CACHE =
            Collections.synchronizedMap(new WeakHashMap<>());

    private EnvironmentOps() {}

    static TensorOps resolve(Environment environment) {
        OpsCache cache = CACHE.computeIfAbsent(environment, OpsCache::new);
        return environment.executionMode() == ExecutionMode.LAZY
                ? cache.lazyOps()
                : cache.eagerOps();
    }

    private static final class OpsCache {
        private final Environment environment;
        private volatile TensorOps eagerOps;
        private volatile TensorOps lazyOps;

        private OpsCache(Environment environment) {
            this.environment = environment;
        }

        private TensorOps eagerOps() {
            TensorOps current = eagerOps;
            if (current != null) {
                return current;
            }
            synchronized (this) {
                if (eagerOps == null) {
                    MemoryDomain<?> domain =
                            environment.runtimeFor(environment.defaultDevice()).memoryDomain();
                    eagerOps = new EagerTensorOps(domain);
                }
                return eagerOps;
            }
        }

        private TensorOps lazyOps() {
            TensorOps current = lazyOps;
            if (current != null) {
                return current;
            }
            synchronized (this) {
                if (lazyOps == null) {
                    lazyOps = new LazyTensorOps();
                }
                return lazyOps;
            }
        }
    }
}
