package ai.qxotic.jota;

import ai.qxotic.jota.memory.MemoryDomain;
import ai.qxotic.jota.runtime.DefaultRuntimeRegistry;
import ai.qxotic.jota.runtime.DeviceRuntime;
import ai.qxotic.jota.runtime.RuntimeDiagnostic;
import ai.qxotic.jota.runtime.RuntimeRegistry;
import ai.qxotic.jota.runtime.panama.PanamaDeviceRuntime;
import ai.qxotic.jota.runtime.spi.DeviceRuntimeProvider;
import ai.qxotic.jota.runtime.spi.RuntimeProbe;
import ai.qxotic.jota.tensor.ComputeEngine;
import java.lang.foreign.MemorySegment;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.ServiceLoader;
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
    private final RuntimeRegistry runtimes;
    private final ExecutionMode executionMode;

    public Environment(
            Device defaultDevice,
            DataType defaultFloat,
            RuntimeRegistry runtimes,
            ExecutionMode executionMode) {
        this.defaultDevice = Objects.requireNonNull(defaultDevice, "defaultDevice");
        this.defaultFloat = Objects.requireNonNull(defaultFloat, "defaultFloat");
        this.runtimes = Objects.requireNonNull(runtimes, "runtimes");
        this.executionMode = Objects.requireNonNull(executionMode, "executionMode");
    }

    public static Environment current() {
        return CURRENT.isBound() ? CURRENT.get() : global();
    }

    @SuppressWarnings("unchecked")
    public MemoryDomain<MemorySegment> nativeMemoryDomain() {
        return (MemoryDomain<MemorySegment>) runtimes.nativeRuntime().memoryDomain();
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
        return runtimes;
    }

    public DeviceRuntime runtimeFor(Device device) {
        return runtimes.runtimeFor(device);
    }

    public DeviceRuntime nativeRuntime() {
        return runtimes.nativeRuntime();
    }

    public ComputeEngine computeEngineFor(Device device) {
        return runtimes.runtimeFor(device).computeEngine();
    }

    public ExecutionMode executionMode() {
        return executionMode;
    }

    public List<RuntimeDiagnostic> runtimeDiagnostics() {
        return runtimes.diagnostics();
    }

    private static RuntimeRegistry buildDefaultRuntimes() {
        DefaultRuntimeRegistry registry =
                DefaultRuntimeRegistry.withNative(new PanamaDeviceRuntime());
        ServiceLoader<DeviceRuntimeProvider> providers =
                ServiceLoader.load(DeviceRuntimeProvider.class);
        providers.stream()
                .map(ServiceLoader.Provider::get)
                .sorted(Comparator.comparingInt(DeviceRuntimeProvider::priority).reversed())
                .forEach(provider -> registerProvider(registry, provider));
        return registry;
    }

    private static void registerProvider(
            DefaultRuntimeRegistry registry, DeviceRuntimeProvider provider) {
        RuntimeProbe probe;
        try {
            probe = provider.probe();
        } catch (Throwable t) {
            probe = RuntimeProbe.error("Runtime probe threw unexpectedly", t);
        }
        registry.addDiagnostic(new RuntimeDiagnostic(provider.id(), provider.device(), probe));
        if (!probe.isAvailable()) {
            return;
        }
        if (provider.device().equals(Device.PANAMA)) {
            registry.addDiagnostic(
                    new RuntimeDiagnostic(
                            provider.id(),
                            provider.device(),
                            RuntimeProbe.misconfigured(
                                    "Optional provider targets reserved native device",
                                    "Do not register external providers for Device.PANAMA",
                                    null)));
            return;
        }
        try {
            registry.register(provider.create());
        } catch (Throwable t) {
            registry.addDiagnostic(
                    new RuntimeDiagnostic(
                            provider.id(),
                            provider.device(),
                            RuntimeProbe.error("Runtime provider failed to create runtime", t)));
        }
    }
}
