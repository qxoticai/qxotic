package ai.qxotic.jota.backend;

import ai.qxotic.jota.Device;
import ai.qxotic.jota.DeviceRegistry;
import ai.qxotic.jota.memory.MemoryContext;
import ai.qxotic.jota.tensor.ComputeEngine;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

public final class DefaultBackendRegistry implements BackendRegistry {

    private final Map<Device, Backend> backends = new ConcurrentHashMap<>();
    private volatile Backend nativeBackend;

    public static DefaultBackendRegistry withNative(Backend backend) {
        DefaultBackendRegistry registry = new DefaultBackendRegistry();
        registry.registerNative(backend);
        return registry;
    }

    public static DefaultBackendRegistry fromDeviceRegistry(DeviceRegistry registry) {
        Objects.requireNonNull(registry, "registry");
        DefaultBackendRegistry backends = new DefaultBackendRegistry();
        for (Device device : registry.devices()) {
            MemoryContext<?> context = registry.context(device);
            ComputeEngine engine = registry.engine(device);
            backends.register(new LegacyBackend(context, engine));
        }
        if (backends.nativeBackend == null && backends.hasBackend(Device.PANAMA)) {
            backends.registerNative(backends.backend(Device.PANAMA));
        }
        if (backends.nativeBackend == null && !backends.backends.isEmpty()) {
            backends.registerNative(backends.backends.values().iterator().next());
        }
        return backends;
    }

    @Override
    public void register(Backend backend) {
        Objects.requireNonNull(backend, "backend");
        backends.put(backend.device(), backend);
        if (nativeBackend == null && backend.device().equals(Device.PANAMA)) {
            nativeBackend = backend;
        }
    }

    public void registerNative(Backend backend) {
        register(backend);
        nativeBackend = backend;
    }

    @Override
    public Backend backend(Device device) {
        Backend backend = backends.get(device);
        if (backend == null) {
            throw new IllegalStateException("No backend registered for " + device);
        }
        return backend;
    }

    @Override
    public boolean hasBackend(Device device) {
        return backends.containsKey(device);
    }

    @Override
    public Backend nativeBackend() {
        Backend backend = nativeBackend;
        if (backend == null) {
            throw new IllegalStateException("No native backend registered");
        }
        return backend;
    }

    @Override
    public Set<Device> devices() {
        return Set.copyOf(backends.keySet());
    }

    private static final class LegacyBackend implements Backend {
        private final MemoryContext<?> context;
        private final ComputeEngine engine;

        private LegacyBackend(MemoryContext<?> context, ComputeEngine engine) {
            this.context = Objects.requireNonNull(context, "context");
            this.engine = Objects.requireNonNull(engine, "engine");
        }

        @Override
        public Device device() {
            return context.device();
        }

        @Override
        public MemoryContext<?> memoryContext() {
            return context;
        }

        @Override
        public ComputeEngine computeEngine() {
            return engine;
        }

        @Override
        public KernelPipeline kernelPipeline() {
            return null;
        }
    }
}
