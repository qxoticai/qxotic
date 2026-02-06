package ai.qxotic.jota;

import ai.qxotic.jota.c.CDeviceRuntime;
import ai.qxotic.jota.c.CNative;
import ai.qxotic.jota.hip.HipDeviceRuntime;
import ai.qxotic.jota.hip.HipRuntime;
import ai.qxotic.jota.memory.MemoryDomain;
import ai.qxotic.jota.panama.PanamaDeviceRuntime;
import ai.qxotic.jota.runtime.DefaultRuntimeRegistry;
import ai.qxotic.jota.runtime.DeviceRuntime;
import ai.qxotic.jota.runtime.RuntimeRegistry;
import ai.qxotic.jota.tensor.ComputeEngine;
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
                    buildDefaultRuntimes(),
                    ExecutionMode.LAZY);

    private final Device defaultDevice;
    private final DataType defaultFloat;
    private final RuntimeRegistry backends;
    private final ExecutionMode executionMode;

    public Environment(
            Device defaultDevice,
            DataType defaultFloat,
            RuntimeRegistry runtimes,
            ExecutionMode executionMode) {
        this.defaultDevice = Objects.requireNonNull(defaultDevice, "defaultDevice");
        this.defaultFloat = Objects.requireNonNull(defaultFloat, "defaultFloat");
        this.backends = Objects.requireNonNull(runtimes, "runtimes");
        this.executionMode = Objects.requireNonNull(executionMode, "executionMode");
    }

    public static Environment current() {
        return CURRENT.isBound() ? CURRENT.get() : global();
    }

    @SuppressWarnings("unchecked")
    public MemoryDomain<MemorySegment> nativeMemoryDomain() {
        return (MemoryDomain<MemorySegment>) backends.nativeRuntime().memoryDomain();
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

    public RuntimeRegistry runtimes() {
        return backends;
    }

    public DeviceRuntime runtimeFor(Device device) {
        return backends.runtimeFor(device);
    }

    public DeviceRuntime nativeRuntime() {
        return backends.nativeRuntime();
    }

    public ComputeEngine computeEngineFor(Device device) {
        return backends.runtimeFor(device).computeEngine();
    }

    public ExecutionMode executionMode() {
        return executionMode;
    }

    private static RuntimeRegistry buildDefaultRuntimes() {
        DefaultRuntimeRegistry registry =
                DefaultRuntimeRegistry.withNative(new PanamaDeviceRuntime());
        if (CNative.isAvailable()) {
            registry.register(new CDeviceRuntime());
        }
        if (HipRuntime.isAvailable()) {
            registry.register(new HipDeviceRuntime());
        }
        return registry;
    }
}
