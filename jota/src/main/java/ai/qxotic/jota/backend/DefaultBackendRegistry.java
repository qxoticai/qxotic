package ai.qxotic.jota.backend;

import ai.qxotic.jota.Device;
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
}
