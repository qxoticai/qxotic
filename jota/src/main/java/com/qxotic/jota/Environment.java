package com.qxotic.jota;

import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.runtime.DefaultRuntimeRegistry;
import com.qxotic.jota.runtime.DeviceRuntime;
import com.qxotic.jota.runtime.RuntimeDiagnostic;
import com.qxotic.jota.runtime.RuntimeRegistry;
import com.qxotic.jota.runtime.panama.PanamaDeviceRuntime;
import com.qxotic.jota.runtime.spi.DeviceRuntimeProvider;
import com.qxotic.jota.runtime.spi.RuntimeProbe;
import com.qxotic.jota.tensor.ComputeEngine;
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

    private final Device defaultDevice;
    private final DataType defaultFloat;
    private final RuntimeRegistry runtimes;

    /**
     * Creates an immutable runtime environment.
     *
     * <p>Invariants:
     *
     * <ul>
     *   <li>{@code defaultFloat} must be floating-point
     *   <li>{@code runtimes} must contain a runtime for {@code defaultDevice}
     * </ul>
     */
    public Environment(Device defaultDevice, DataType defaultFloat, RuntimeRegistry runtimes) {
        this.defaultDevice = Objects.requireNonNull(defaultDevice, "defaultDevice");
        this.defaultFloat = Objects.requireNonNull(defaultFloat, "defaultFloat");
        this.runtimes = Objects.requireNonNull(runtimes, "runtimes");
        if (!defaultFloat.isFloatingPoint()) {
            throw new IllegalArgumentException(
                    "defaultFloat must be floating-point, got " + defaultFloat);
        }
        if (!runtimes.hasRuntime(defaultDevice)) {
            throw new IllegalArgumentException(
                    "No runtime registered for defaultDevice " + defaultDevice);
        }
    }

    /**
     * Returns the currently active environment.
     *
     * <p>If a scoped environment is active via {@link #with(Environment, Supplier)}, that scoped
     * environment is returned. Otherwise the globally configured environment is returned.
     */
    public static Environment current() {
        return CURRENT.isBound() ? CURRENT.get() : global();
    }

    @SuppressWarnings("unchecked")
    public MemoryDomain<MemorySegment> nativeMemoryDomain() {
        return (MemoryDomain<MemorySegment>) runtimes.nativeRuntime().memoryDomain();
    }

    @SuppressWarnings("unchecked")
    public <B> MemoryDomain<B> memoryDomainFor(Device device) {
        Objects.requireNonNull(device, "device");
        return (MemoryDomain<B>) runtimeFor(device).memoryDomain();
    }

    /** Returns the globally configured environment, or the default built-in global environment. */
    public static Environment global() {
        Environment configured = GLOBAL.get();
        return configured == null ? DefaultGlobalHolder.INSTANCE : configured;
    }

    /**
     * Configures the process-wide global environment.
     *
     * <p>This method can only be called once. Subsequent calls fail.
     */
    public static void configureGlobal(Environment environment) {
        Objects.requireNonNull(environment, "environment");
        if (!GLOBAL.compareAndSet(null, environment)) {
            throw new IllegalStateException("Global Environment already configured");
        }
    }

    /**
     * Executes {@code action} with {@code environment} bound as the current scoped environment.
     *
     * <p>The binding applies only within this call and does not mutate global configuration.
     */
    public static <T> T with(Environment environment, Supplier<T> action) {
        Objects.requireNonNull(environment, "environment");
        Objects.requireNonNull(action, "action");
        return ScopedValue.where(CURRENT, environment).call(action::get);
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

    private static final class DefaultGlobalHolder {
        private static final Environment INSTANCE =
                new Environment(
                        Device.PANAMA, DataTypeImpl.defaultFloatValue(), buildDefaultRuntimes());
    }
}
