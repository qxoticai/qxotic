package ai.qxotic.jota;

import ai.qxotic.jota.memory.MemoryContext;
import ai.qxotic.jota.memory.impl.ContextFactory;
import ai.qxotic.jota.tensor.ComputeEngine;
import ai.qxotic.jota.tensor.JavaComputeEngine;
import java.lang.foreign.MemorySegment;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

public final class DeviceRegistry {

    private static final MemoryContext<MemorySegment> DEFAULT_CONTEXT =
            ContextFactory.ofMemorySegment();
    private static final DeviceRegistry GLOBAL =
            builder().register(DEFAULT_CONTEXT, new JavaComputeEngine(DEFAULT_CONTEXT)).build();

    private final Map<Device, MemoryContext<?>> contexts;
    private final Map<Device, ComputeEngine> engines;

    private DeviceRegistry(Builder builder) {
        this.contexts = Map.copyOf(builder.contexts);
        this.engines = Map.copyOf(builder.engines);
    }

    public static DeviceRegistry global() {
        return GLOBAL;
    }

    public static Builder builder() {
        return new Builder();
    }

    public MemoryContext<?> context(Device device) {
        MemoryContext<?> context = contexts.get(device);
        if (context == null) {
            throw new IllegalStateException("No MemoryContext registered for " + device);
        }
        return context;
    }

    public ComputeEngine engine(Device device) {
        ComputeEngine engine = engines.get(device);
        if (engine == null) {
            throw new IllegalStateException("No ComputeEngine registered for " + device);
        }
        return engine;
    }

    public Set<Device> devices() {
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

        public DeviceRegistry build() {
            return new DeviceRegistry(this);
        }
    }
}
