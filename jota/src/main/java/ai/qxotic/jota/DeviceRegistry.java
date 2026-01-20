package ai.qxotic.jota;

import ai.qxotic.jota.memory.MemoryContext;
import ai.qxotic.jota.tensor.ComputeEngine;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

public final class DeviceRegistry {

    private static final ConcurrentMap<Device, MemoryContext<?>> CONTEXTS =
            new ConcurrentHashMap<>();
    private static final ConcurrentMap<Device, ComputeEngine> ENGINES = new ConcurrentHashMap<>();

    private DeviceRegistry() {}

    public static void register(Device device, MemoryContext<?> context, ComputeEngine engine) {
        Objects.requireNonNull(device, "device");
        Objects.requireNonNull(context, "context");
        Objects.requireNonNull(engine, "engine");

        MemoryContext<?> existingContext = CONTEXTS.putIfAbsent(device, context);
        if (existingContext != null && existingContext != context) {
            throw new IllegalStateException("MemoryContext already registered for " + device);
        }
        ComputeEngine existingEngine = ENGINES.putIfAbsent(device, engine);
        if (existingEngine != null && existingEngine != engine) {
            throw new IllegalStateException("ComputeEngine already registered for " + device);
        }
    }

    public static MemoryContext<?> context(Device device) {
        MemoryContext<?> context = CONTEXTS.get(device);
        if (context == null) {
            throw new IllegalStateException("No MemoryContext registered for " + device);
        }
        return context;
    }

    public static ComputeEngine engine(Device device) {
        ComputeEngine engine = ENGINES.get(device);
        if (engine == null) {
            throw new IllegalStateException("No ComputeEngine registered for " + device);
        }
        return engine;
    }
}
