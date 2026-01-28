package ai.qxotic.jota;

import ai.qxotic.jota.backend.Backend;
import ai.qxotic.jota.backend.BackendRegistry;
import ai.qxotic.jota.backend.DefaultBackendRegistry;
import ai.qxotic.jota.hip.HipBackend;
import ai.qxotic.jota.hip.HipRuntime;
import ai.qxotic.jota.memory.MemoryContext;
import ai.qxotic.jota.panama.PanamaBackend;
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
                    buildDefaultBackends(),
                    ExecutionMode.LAZY);

    private final Device defaultDevice;
    private final DataType defaultFloat;
    private final BackendRegistry backends;
    private final ExecutionMode executionMode;

    public Environment(
            Device defaultDevice,
            DataType defaultFloat,
            BackendRegistry backends,
            ExecutionMode executionMode) {
        this.defaultDevice = Objects.requireNonNull(defaultDevice, "defaultDevice");
        this.defaultFloat = Objects.requireNonNull(defaultFloat, "defaultFloat");
        this.backends = Objects.requireNonNull(backends, "backends");
        this.executionMode = Objects.requireNonNull(executionMode, "executionMode");
    }

    public static Environment current() {
        return CURRENT.isBound() ? CURRENT.get() : global();
    }

    @SuppressWarnings("unchecked")
    public MemoryContext<MemorySegment> panamaContext() {
        return (MemoryContext<MemorySegment>) backends.nativeBackend().memoryContext();
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

    public BackendRegistry backends() {
        return backends;
    }

    public Backend backend(Device device) {
        return backends.backend(device);
    }

    public Backend nativeBackend() {
        return backends.nativeBackend();
    }

    public ComputeEngine engineFor(Device device) {
        return backends.backend(device).computeEngine();
    }

    public ExecutionMode executionMode() {
        return executionMode;
    }

    private static BackendRegistry buildDefaultBackends() {
        DefaultBackendRegistry registry = DefaultBackendRegistry.withNative(new PanamaBackend());
        if (HipRuntime.isAvailable()) {
            registry.register(new HipBackend());
        }
        //        if (WebGPUSupport.hasGpuAdapter()) {
        //            registry.register(WebGPUWBackend.create());
        //        }
        return registry;
    }
}
