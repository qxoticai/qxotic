package ai.qxotic.jota;

import ai.qxotic.jota.backend.Backend;
import ai.qxotic.jota.backend.BackendRegistry;
import ai.qxotic.jota.backend.DefaultBackendRegistry;
import ai.qxotic.jota.hip.HipBackend;
import ai.qxotic.jota.hip.HipRuntime;
import ai.qxotic.jota.memory.MemoryContext;
import ai.qxotic.jota.panama.PanamaBackend;
import ai.qxotic.jota.tensor.ComputeEngine;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

public final class DeviceRegistry {

    private static final DeviceRegistry GLOBAL = buildGlobal();

    private final Map<Device, MemoryContext<?>> contexts;
    private final Map<Device, ComputeEngine> engines;
    private final BackendRegistry backends;

    private DeviceRegistry(Builder builder) {
        this.contexts = Map.copyOf(builder.contexts);
        this.engines = Map.copyOf(builder.engines);
        this.backends = null;
    }

    private DeviceRegistry(BackendRegistry backends) {
        this.contexts = Map.of();
        this.engines = Map.of();
        this.backends = Objects.requireNonNull(backends, "backends");
    }

    public static DeviceRegistry global() {
        return GLOBAL;
    }

    private static DeviceRegistry buildGlobal() {
        DefaultBackendRegistry registry =
                DefaultBackendRegistry.withNative(new PanamaBackend());
        if (HipRuntime.isAvailable()) {
            registry.register(new HipBackend());
        }
        return fromBackends(registry);
    }

    public static Builder builder() {
        return new Builder();
    }

    public static DeviceRegistry fromBackends(BackendRegistry backends) {
        return new DeviceRegistry(backends);
    }

    public MemoryContext<?> context(Device device) {
        if (backends != null) {
            return backends.backend(device).memoryContext();
        }
        MemoryContext<?> context = contexts.get(device);
        if (context == null) {
            throw new IllegalStateException("No MemoryContext registered for " + device);
        }
        return context;
    }

    public ComputeEngine engine(Device device) {
        if (backends != null) {
            return backends.backend(device).computeEngine();
        }
        ComputeEngine engine = engines.get(device);
        if (engine == null) {
            throw new IllegalStateException("No ComputeEngine registered for " + device);
        }
        return engine;
    }

    public Set<Device> devices() {
        if (backends != null) {
            return backends.devices();
        }
        return contexts.keySet(); // Already unmodifiable from Map.copyOf
    }

    public static final class Builder {
        private final Map<Device, MemoryContext<?>> contexts = new HashMap<>();
        private final Map<Device, ComputeEngine> engines = new HashMap<>();

        private Builder() {}

        public Builder register(MemoryContext<?> context, ComputeEngine engine) {
            Objects.requireNonNull(context, "context");
            Objects.requireNonNull(engine, "engine");

            Device device = context.device();
            MemoryContext<?> existingContext = contexts.putIfAbsent(device, context);
            if (existingContext != null && existingContext != context) {
                throw new IllegalStateException("MemoryContext already registered for " + device);
            }
            ComputeEngine existingEngine = engines.putIfAbsent(device, engine);
            if (existingEngine != null && existingEngine != engine) {
                throw new IllegalStateException("ComputeEngine already registered for " + device);
            }
            return this;
        }

        public Builder register(Backend backend) {
            Objects.requireNonNull(backend, "backend");
            return register(backend.memoryContext(), backend.computeEngine());
        }

        public DeviceRegistry build() {
            return new DeviceRegistry(this);
        }
    }
}
