package com.qxotic.jota.tensor;

import com.qxotic.jota.Environment;
import com.qxotic.jota.ExecutionMode;
import com.qxotic.jota.runtime.DeviceRuntime;
import java.util.Collections;
import java.util.Map;
import java.util.Objects;
import java.util.WeakHashMap;
import java.util.function.Supplier;

public final class TensorOpsContext {

    private static final ScopedValue<TensorOps> CURRENT = ScopedValue.newInstance();

    private static final Map<Environment, OpsCache> CACHE =
            Collections.synchronizedMap(new WeakHashMap<>());

    private TensorOpsContext() {}

    public static TensorOps current() {
        return CURRENT.isBound() ? CURRENT.get() : null;
    }

    public static TensorOps require() {
        if (CURRENT.isBound()) {
            return CURRENT.get();
        }
        return resolveOps(Environment.current());
    }

    private static TensorOps resolveOps(Environment environment) {
        OpsCache cache = CACHE.computeIfAbsent(environment, OpsCache::new);
        return environment.executionMode() == ExecutionMode.LAZY
                ? cache.lazyOps()
                : cache.eagerOps();
    }

    public static <T> T withIR(Supplier<T> supplier) {
        return with(new IRTensorOps(), supplier);
    }

    public static <T> T with(TensorOps ops, Supplier<T> action) {
        Objects.requireNonNull(ops, "ops");
        Objects.requireNonNull(action, "action");
        try {
            return ScopedValue.where(CURRENT, ops).call(action::get);
        } catch (RuntimeException e) {
            throw e;
        } catch (Exception e) {
            throw new IllegalStateException("Scoped TensorOps action failed", e);
        }
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
                    DeviceRuntime runtime = environment.runtimeFor(environment.defaultDevice());
                    eagerOps = new EagerTensorOps(runtime);
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
