package ai.qxotic.jota;

import ai.qxotic.jota.memory.MemoryContext;
import java.lang.foreign.MemorySegment;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Supplier;

public final class Environment {

    private static final ScopedValue<Environment> CURRENT = ScopedValue.newInstance();
    private static final AtomicReference<Environment> GLOBAL = new AtomicReference<>();
    private static final Environment DEFAULT_GLOBAL =
            new Environment(
                    Device.PANAMA,
                    DataTypeImpl.defaultFloatValue(),
                    DeviceRegistry.global(),
                    ExecutionMode.EAGER);

    private final Device defaultDevice;
    private final DataType defaultFloat;
    private final DeviceRegistry registry;
    private final ExecutionMode executionMode;

    public Environment(Device defaultDevice, DataType defaultFloat, DeviceRegistry registry) {
        this(defaultDevice, defaultFloat, registry, ExecutionMode.EAGER);
    }

    public Environment(
            Device defaultDevice,
            DataType defaultFloat,
            DeviceRegistry registry,
            ExecutionMode executionMode) {
        this.defaultDevice = Objects.requireNonNull(defaultDevice, "defaultDevice");
        this.defaultFloat = Objects.requireNonNull(defaultFloat, "defaultFloat");
        this.registry = Objects.requireNonNull(registry, "registry");
        this.executionMode = Objects.requireNonNull(executionMode, "executionMode");
    }

    public static Environment current() {
        return CURRENT.isBound() ? CURRENT.get() : global();
    }

    @SuppressWarnings("unchecked")
    public MemoryContext<MemorySegment> panamaContext() {
        return (MemoryContext<MemorySegment>) registry.context(Device.PANAMA);
    }

    public static Environment global() {
        Environment configured = GLOBAL.get();
        return configured == null ? DEFAULT_GLOBAL : configured;
    }

    public static void configureGlobal(Environment environment) {
        Objects.requireNonNull(environment, "environment");
        if (!GLOBAL.compareAndSet(null, environment)) {
            throw new IllegalStateException("Global Environment already configured");
        }
    }

    public static <T> T with(Environment environment, Supplier<T> action) {
        Objects.requireNonNull(environment, "environment");
        Objects.requireNonNull(action, "action");
        try {
            return ScopedValue.where(CURRENT, environment).call(action::get);
        } catch (RuntimeException e) {
            throw e;
        } catch (Exception e) {
            throw new IllegalStateException("Scoped Environment action failed", e);
        }
    }

    public Device defaultDevice() {
        return defaultDevice;
    }

    public DataType defaultFloat() {
        return defaultFloat;
    }

    public DeviceRegistry registry() {
        return registry;
    }

    public ExecutionMode executionMode() {
        return executionMode;
    }
}
