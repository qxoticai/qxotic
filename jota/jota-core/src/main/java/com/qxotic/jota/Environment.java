package com.qxotic.jota;

import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.runtime.ComputeEngine;
import com.qxotic.jota.runtime.DefaultRuntimeRegistry;
import com.qxotic.jota.runtime.DeviceRuntime;
import com.qxotic.jota.runtime.RuntimeDiagnostic;
import com.qxotic.jota.runtime.RuntimeRegistry;
import com.qxotic.jota.runtime.panama.PanamaDeviceRuntime;
import com.qxotic.jota.runtime.spi.DeviceRuntimeProvider;
import com.qxotic.jota.runtime.spi.RuntimeProbe;
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
    private static final boolean LOG_RUNTIME_WARNINGS =
            Boolean.parseBoolean(System.getProperty("jota.runtime.probe.log", "true"));

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
        return (MemoryDomain<MemorySegment>) runtimeFor(Device.NATIVE).memoryDomain();
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
        DefaultRuntimeRegistry registry = new DefaultRuntimeRegistry();
        DeviceRuntime panamaRuntime = new PanamaDeviceRuntime();
        registry.register(panamaRuntime);
        registry.registerNative(panamaRuntime);
        ServiceLoader<DeviceRuntimeProvider> providers =
                ServiceLoader.load(DeviceRuntimeProvider.class);
        providers.stream()
                .map(ServiceLoader.Provider::get)
                .sorted(Comparator.comparingInt(DeviceRuntimeProvider::priority).reversed())
                .forEach(provider -> registerProvider(registry, provider));
        registry.registerNative(registry.runtimeFor(selectNativeBackend(registry)));
        return registry;
    }

    private static Device selectNativeBackend(RuntimeRegistry registry) {
        Device override = parseNativeBackendOverride(System.getProperty("jota.native.backend"));
        if (override != null) {
            if (!registry.hasRuntime(override)) {
                throw new IllegalStateException(
                        "Configured jota.native.backend selects unavailable backend: "
                                + override
                                + ". Available devices: "
                                + registry.devices());
            }
            return override;
        }
        if (isNativeImageRuntime() && registry.hasRuntime(Device.C)) {
            return Device.C;
        }
        return Device.PANAMA;
    }

    private static Device parseNativeBackendOverride(String rawValue) {
        if (rawValue == null || rawValue.isBlank()) {
            return null;
        }
        String value = rawValue.trim().toLowerCase();
        return switch (value) {
            case "auto", "native" -> null;
            case "panama", "java", "jvm", "ffm" -> Device.PANAMA;
            case "c" -> Device.C;
            default ->
                    throw new IllegalArgumentException(
                            "Unsupported jota.native.backend='"
                                    + rawValue
                                    + "'. Supported values: auto, native, panama, java, c");
        };
    }

    private static boolean isNativeImageRuntime() {
        String imageCode = System.getProperty("org.graalvm.nativeimage.imagecode");
        return imageCode != null && !imageCode.isBlank();
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
            logUnavailableRuntime(provider, probe);
            return;
        }
        if (provider.device().equals(Device.PANAMA) || provider.device().equals(Device.NATIVE)) {
            registry.addDiagnostic(
                    new RuntimeDiagnostic(
                            provider.id(),
                            provider.device(),
                            RuntimeProbe.misconfigured(
                                    "Optional provider targets reserved native device",
                                    "Do not register external providers for"
                                            + " Device.PANAMA/Device.NATIVE",
                                    null)));
            logUnavailableRuntime(
                    provider,
                    RuntimeProbe.misconfigured(
                            "Optional provider targets reserved native device",
                            "Do not register external providers for Device.PANAMA/Device.NATIVE",
                            null));
            return;
        }
        try {
            registry.register(provider.create());
        } catch (Throwable t) {
            RuntimeProbe failure =
                    RuntimeProbe.error("Runtime provider failed to create runtime", t);
            registry.addDiagnostic(
                    new RuntimeDiagnostic(provider.id(), provider.device(), failure));
            logUnavailableRuntime(provider, failure);
        }
    }

    private static void logUnavailableRuntime(DeviceRuntimeProvider provider, RuntimeProbe probe) {
        if (!LOG_RUNTIME_WARNINGS) {
            return;
        }
        StringBuilder warning = new StringBuilder();
        warning.append("[jota-runtime] WARNING: backend '")
                .append(provider.id())
                .append("' on device ")
                .append(provider.device().leafName())
                .append(" is unavailable: ")
                .append(probe.message());
        if (probe.hint() != null) {
            warning.append(" | hint: ").append(probe.hint());
        }
        if (probe.cause() != null) {
            warning.append(" | cause: ")
                    .append(probe.cause().getClass().getSimpleName())
                    .append(": ")
                    .append(probe.cause().getMessage());
        }
        System.err.println(warning);
    }

    private static final class DefaultGlobalHolder {
        private static final Environment INSTANCE =
                new Environment(
                        Device.NATIVE, DataTypeImpl.defaultFloatValue(), buildDefaultRuntimes());
    }
}
