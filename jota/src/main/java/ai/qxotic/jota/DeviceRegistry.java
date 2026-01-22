package ai.qxotic.jota;

import ai.qxotic.jota.memory.MemoryContext;
import ai.qxotic.jota.tensor.ComputeEngine;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

public final class DeviceRegistry {

    private static final DeviceRegistry GLOBAL = new DeviceRegistry();

    private final ConcurrentMap<Device, MemoryContext<?>> contexts = new ConcurrentHashMap<>();
    private final ConcurrentMap<Device, ComputeEngine> engines = new ConcurrentHashMap<>();

    public static DeviceRegistry global() {
        return GLOBAL;
    }

    public void register(Device device, MemoryContext<?> context, ComputeEngine engine) {
        registerInstance(device, context, engine);
    }

    public MemoryContext<?> context(Device device) {
        return contextInstance(device);
    }

    public ComputeEngine engine(Device device) {
        return engineInstance(device);
    }

    public Set<Device> devices() {
        return devicesInstance();
    }

    private void registerInstance(Device device, MemoryContext<?> context, ComputeEngine engine) {
        Objects.requireNonNull(device, "device");
        Objects.requireNonNull(context, "context");
        Objects.requireNonNull(engine, "engine");

        MemoryContext<?> existingContext = contexts.putIfAbsent(device, context);
        if (existingContext != null && existingContext != context) {
            throw new IllegalStateException("MemoryContext already registered for " + device);
        }
        ComputeEngine existingEngine = engines.putIfAbsent(device, engine);
        if (existingEngine != null && existingEngine != engine) {
            throw new IllegalStateException("ComputeEngine already registered for " + device);
        }
    }

    private MemoryContext<?> contextInstance(Device device) {
        MemoryContext<?> context = contexts.get(device);
        if (context == null) {
            throw new IllegalStateException("No MemoryContext registered for " + device);
        }
        return context;
    }

    private ComputeEngine engineInstance(Device device) {
        ComputeEngine engine = engines.get(device);
        if (engine == null) {
            throw new IllegalStateException("No ComputeEngine registered for " + device);
        }
        return engine;
    }

    private Set<Device> devicesInstance() {
        return Set.copyOf(contexts.keySet());
    }
}
