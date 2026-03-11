package com.qxotic.jota;

import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.runtime.ComputeEngine;
import com.qxotic.jota.runtime.DefaultRuntimeRegistry;
import com.qxotic.jota.runtime.DeviceRuntime;
import com.qxotic.jota.runtime.RuntimeDiagnostic;
import com.qxotic.jota.runtime.RuntimeRegistry;
import com.qxotic.jota.runtime.spi.DeviceRuntimeProvider;
import com.qxotic.jota.runtime.spi.RuntimeProbe;
import java.lang.foreign.MemorySegment;
import java.util.Comparator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.ServiceLoader;
import java.util.Set;
import java.util.StringJoiner;
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
        Set<String> includedBackends =
                parseBackendList(System.getProperty("jota.backends.include"));
        Set<String> excludedBackends =
                parseBackendList(System.getProperty("jota.backends.exclude"));
        ServiceLoader<DeviceRuntimeProvider> providers =
                ServiceLoader.load(DeviceRuntimeProvider.class);
        providers.stream()
                .map(ServiceLoader.Provider::get)
                .sorted(Comparator.comparingInt(DeviceRuntimeProvider::priority).reversed())
                .forEach(
                        provider ->
                                registerProvider(
                                        registry, provider, includedBackends, excludedBackends));
        Device configuredOverride =
                parseNativeBackendOverride(System.getProperty("jota.native.backend"));
        Device nativeBackend = selectNativeBackend(registry);
        if (!registry.hasRuntime(nativeBackend)) {
            throw missingNativeRuntimeException(
                    registry, "Selected native backend " + nativeBackend + " is unavailable");
        }
        DeviceRuntime selectedRuntime = registry.runtimeFor(nativeBackend);
        if (!selectedRuntime.supportsNativeRuntimeAlias()) {
            throw missingNativeRuntimeException(
                    registry,
                    "Selected native backend "
                            + nativeBackend
                            + " does not support Device.NATIVE alias");
        }
        registry.registerNative(selectedRuntime);
        return registry;
    }

    private static Device selectNativeBackend(RuntimeRegistry registry) {
        Device override = parseNativeBackendOverride(System.getProperty("jota.native.backend"));
        if (override != null) {
            if (!registry.hasRuntime(override)) {
                throw missingNativeRuntimeException(
                        registry,
                        "Configured jota.native.backend selects unavailable backend: " + override);
            }
            return override;
        }
        if (isNativeImageRuntime() && registry.hasRuntime(Device.C)) {
            return Device.C;
        }
        if (!isNativeImageRuntime() && registry.hasRuntime(Device.PANAMA)) {
            return Device.PANAMA;
        }
        Device compatibleFallback = findCompatibleNativeBackend(registry);
        if (compatibleFallback != null) {
            return compatibleFallback;
        }
        throw missingNativeRuntimeException(
                registry,
                "No compatible runtime available for Device.NATIVE"
                        + (isNativeImageRuntime()
                                ? " in GraalVM Native Image"
                                : " in JVM runtime"));
    }

    private static Device parseNativeBackendOverride(String rawValue) {
        if (rawValue == null || rawValue.isBlank()) {
            return null;
        }
        String value = rawValue.trim().toLowerCase();
        return switch (value) {
            case "auto", "native" -> null;
            case "panama", "jvm", "ffm" -> Device.PANAMA;
            case "c" -> Device.C;
            case "hip" -> Device.HIP;
            case "mojo" -> Device.MOJO;
            case "opencl" -> Device.OPENCL;
            case "metal" -> Device.METAL;
            default ->
                    throw new IllegalArgumentException(
                            "Unsupported jota.native.backend='"
                                    + rawValue
                                    + "'. Supported values: auto, native, panama, c, hip, mojo,"
                                    + " opencl, metal");
        };
    }

    private static Set<String> parseBackendList(String rawValue) {
        if (rawValue == null || rawValue.isBlank()) {
            return Set.of();
        }
        Set<String> values = new LinkedHashSet<>();
        for (String token : rawValue.split("[,\\s]+")) {
            if (!token.isBlank()) {
                values.add(token.trim().toLowerCase(Locale.ROOT));
            }
        }
        return values;
    }

    private static boolean matchesBackendSelector(
            DeviceRuntimeProvider provider, Set<String> selectors) {
        String id = provider.id().toLowerCase(Locale.ROOT);
        String leaf = provider.device().leafName().toLowerCase(Locale.ROOT);
        String full = provider.device().name().toLowerCase(Locale.ROOT);
        return selectors.contains(id) || selectors.contains(leaf) || selectors.contains(full);
    }

    private static RuntimeProbe backendFilterProbe(
            DeviceRuntimeProvider provider,
            Set<String> includedBackends,
            Set<String> excludedBackends) {
        boolean included =
                includedBackends.isEmpty() || matchesBackendSelector(provider, includedBackends);
        boolean excluded =
                !excludedBackends.isEmpty() && matchesBackendSelector(provider, excludedBackends);
        if (!included) {
            return RuntimeProbe.misconfigured(
                    "Runtime provider filtered out by include list",
                    "Add backend to -Djota.backends.include or clear include filter",
                    null);
        }
        if (excluded) {
            return RuntimeProbe.misconfigured(
                    "Runtime provider excluded by configuration",
                    "Remove it from -Djota.backends.exclude to enable (exclude wins over include)",
                    null);
        }
        return null;
    }

    private static boolean isNativeImageRuntime() {
        String imageCode = System.getProperty("org.graalvm.nativeimage.imagecode");
        return imageCode != null && !imageCode.isBlank();
    }

    private static Device findCompatibleNativeBackend(RuntimeRegistry registry) {
        for (Device device : registry.devices()) {
            if (device.equals(Device.NATIVE)) {
                continue;
            }
            if (registry.runtimeFor(device).supportsNativeRuntimeAlias()) {
                return device;
            }
        }
        return null;
    }

    private static IllegalStateException missingNativeRuntimeException(
            RuntimeRegistry registry, String reason) {
        StringBuilder message = new StringBuilder();
        message.append("Unable to configure Device.NATIVE runtime. ").append(reason).append('.');
        message.append('\n')
                .append("Runtime mode: ")
                .append(isNativeImageRuntime() ? "graal-native-image" : "jvm");

        String backendOverride = System.getProperty("jota.native.backend");
        message.append('\n')
                .append("jota.native.backend: ")
                .append(
                        (backendOverride == null || backendOverride.isBlank())
                                ? "<auto>"
                                : backendOverride);

        String includedBackends = System.getProperty("jota.backends.include");
        message.append('\n')
                .append("jota.backends.include: ")
                .append(
                        (includedBackends == null || includedBackends.isBlank())
                                ? "<all>"
                                : includedBackends);

        String excludedBackends = System.getProperty("jota.backends.exclude");
        message.append('\n')
                .append("jota.backends.exclude: ")
                .append(
                        (excludedBackends == null || excludedBackends.isBlank())
                                ? "<none>"
                                : excludedBackends);

        message.append('\n')
                .append("Registered runtimes: ")
                .append(formatDevices(registry.devices()));
        message.append('\n')
                .append("Native-compatible runtimes: ")
                .append(formatCompatibleNativeDevices(registry));

        List<RuntimeDiagnostic> diagnostics = registry.diagnostics();
        if (!diagnostics.isEmpty()) {
            message.append('\n').append("Runtime probe diagnostics:");
            for (RuntimeDiagnostic diagnostic : diagnostics) {
                RuntimeProbe probe = diagnostic.probe();
                message.append('\n')
                        .append("- ")
                        .append(diagnostic.providerId())
                        .append(" [")
                        .append(diagnostic.device().leafName())
                        .append("] ")
                        .append(probe.status().name().toLowerCase())
                        .append(": ")
                        .append(probe.message());
                if (probe.hint() != null) {
                    message.append(" | hint: ").append(probe.hint());
                }
                if (probe.cause() != null) {
                    message.append(" | cause: ")
                            .append(probe.cause().getClass().getSimpleName())
                            .append(": ")
                            .append(probe.cause().getMessage());
                }
            }
        }

        message.append('\n').append("Required: at least one runtime supporting Device.NATIVE");
        message.append(" (Panama, C, or another runtime with supportsNativeRuntimeAlias=true).");
        message.append('\n').append("Suggested fixes:");
        message.append('\n').append("- Include a Panama backend: com.qxotic:jota-backend-panama");
        message.append('\n').append("- Include C backend: com.qxotic:jota-backend-c");
        message.append('\n').append("- Graal Native Image: add dependency com.qxotic:jota-graal");
        message.append('\n')
                .append(
                        "- Explicit backend override: -Djota.native.backend=<backend-id>"
                                + " (e.g. panama, c, hip, mojo, opencl, metal)");
        message.append('\n')
                .append("- Ensure required backends are allowed by include/exclude filters");
        message.append('\n')
                .append("- Run with -Djota.runtime.probe.log=true for backend warnings");
        return new IllegalStateException(message.toString());
    }

    private static String formatDevices(Iterable<Device> devices) {
        StringJoiner joiner = new StringJoiner(", ");
        for (Device device : devices) {
            joiner.add(device.leafName());
        }
        String value = joiner.toString();
        return value.isEmpty() ? "<none>" : value;
    }

    private static String formatCompatibleNativeDevices(RuntimeRegistry registry) {
        StringJoiner joiner = new StringJoiner(", ");
        for (Device device : registry.devices()) {
            if (device.equals(Device.NATIVE)) {
                continue;
            }
            if (registry.runtimeFor(device).supportsNativeRuntimeAlias()) {
                joiner.add(device.leafName());
            }
        }
        String value = joiner.toString();
        return value.isEmpty() ? "<none>" : value;
    }

    private static void registerProvider(
            DefaultRuntimeRegistry registry,
            DeviceRuntimeProvider provider,
            Set<String> includedBackends,
            Set<String> excludedBackends) {
        RuntimeProbe filterProbe = backendFilterProbe(provider, includedBackends, excludedBackends);
        if (filterProbe != null) {
            registry.addDiagnostic(
                    new RuntimeDiagnostic(provider.id(), provider.device(), filterProbe));
            return;
        }
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
        if (provider.device().equals(Device.NATIVE)) {
            registry.addDiagnostic(
                    new RuntimeDiagnostic(
                            provider.id(),
                            provider.device(),
                            RuntimeProbe.misconfigured(
                                    "Optional provider targets reserved native device",
                                    "Do not register external providers for Device.NATIVE",
                                    null)));
            logUnavailableRuntime(
                    provider,
                    RuntimeProbe.misconfigured(
                            "Optional provider targets reserved native device",
                            "Do not register external providers for Device.NATIVE",
                            null));
            return;
        }
        if (provider.device().equals(Device.PANAMA) && registry.hasRuntime(Device.PANAMA)) {
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
        logUnavailableRuntime(provider.id(), provider.device(), probe);
    }

    private static void logUnavailableRuntime(String id, Device device, RuntimeProbe probe) {
        if (!LOG_RUNTIME_WARNINGS) {
            return;
        }
        StringBuilder warning = new StringBuilder();
        warning.append("[jota-runtime] WARNING: backend '")
                .append(id)
                .append("' on device ")
                .append(device.leafName())
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
