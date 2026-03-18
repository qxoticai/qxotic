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
        return (MemoryDomain<MemorySegment>) runtimeFor("native").memoryDomain();
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

    public DeviceRuntime runtimeFor(String nameOrAlias) {
        return runtimes.runtimeFor(nameOrAlias);
    }

    public DeviceRuntime nativeRuntime() {
        return runtimeFor("native");
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

        // Wire native alias
        Device nativeBackend = selectNativeBackend(registry);
        registry.registerAlias("native", nativeBackend);
        // Wire default alias
        registry.registerAlias("default", nativeBackend);

        return registry;
    }

    private static Device selectNativeBackend(DefaultRuntimeRegistry registry) {
        Device override = parseNativeBackendOverride(System.getProperty("jota.native.backend"));
        if (override != null) {
            if (!registry.hasRuntime(override)) {
                throw missingNativeRuntimeException(
                        registry,
                        "Configured jota.native.backend selects unavailable backend: " + override);
            }
            DeviceRuntime runtime = registry.runtimeFor(override);
            if (!runtime.supportsNativeRuntimeAlias()) {
                throw missingNativeRuntimeException(
                        registry,
                        "Selected native backend "
                                + override
                                + " does not support native runtime alias");
            }
            return override;
        }
        Device panama0 = new Device(DeviceType.PANAMA, 0);
        Device c0 = new Device(DeviceType.C, 0);
        if (!isNativeImageRuntime() && registry.hasRuntime(panama0)) {
            return panama0;
        }
        if (isNativeImageRuntime() && registry.hasRuntime(c0)) {
            return c0;
        }
        Device compatibleFallback = findCompatibleNativeBackend(registry);
        if (compatibleFallback != null) {
            return compatibleFallback;
        }
        throw missingNativeRuntimeException(
                registry,
                "No compatible runtime available for native runtime"
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
            case "panama", "jvm", "ffm" -> new Device(DeviceType.PANAMA, 0);
            case "c" -> new Device(DeviceType.C, 0);
            case "hip" -> new Device(DeviceType.HIP, 0);
            case "cuda" -> new Device(DeviceType.CUDA, 0);
            case "mojo" -> new Device(DeviceType.MOJO, 0);
            case "opencl" -> new Device(DeviceType.OPENCL, 0);
            case "metal" -> new Device(DeviceType.METAL, 0);
            default ->
                    throw new IllegalArgumentException(
                            "Unsupported jota.native.backend='"
                                    + rawValue
                                    + "'. Supported values: auto, native, panama, c, hip, cuda,"
                                    + " mojo, opencl, metal");
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
        String runtimeId = provider.deviceType().id();
        return selectors.contains(runtimeId);
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

    private static Device findCompatibleNativeBackend(DefaultRuntimeRegistry registry) {
        for (Device device : registry.devices()) {
            DeviceRuntime runtime = registry.runtimeFor(device);
            if (runtime.supportsNativeRuntimeAlias()) {
                return device;
            }
        }
        return null;
    }

    private static IllegalStateException missingNativeRuntimeException(
            RuntimeRegistry registry, String reason) {
        StringBuilder message = new StringBuilder();
        message.append("Unable to configure native runtime. ").append(reason).append('.');
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

        List<RuntimeDiagnostic> diagnostics = registry.diagnostics();
        if (!diagnostics.isEmpty()) {
            message.append('\n').append("Runtime probe diagnostics:");
            for (RuntimeDiagnostic diagnostic : diagnostics) {
                RuntimeProbe probe = diagnostic.probe();
                message.append('\n')
                        .append("- ")
                        .append(diagnostic.providerId())
                        .append(" [")
                        .append(diagnostic.deviceType().id())
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

        message.append('\n').append("Required: at least one runtime supporting native alias");
        message.append(" (Panama, C, or another runtime with supportsNativeRuntimeAlias=true).");
        message.append('\n').append("Suggested fixes:");
        message.append('\n').append("- Include a Panama backend: com.qxotic:jota-backend-panama");
        message.append('\n').append("- Include C backend: com.qxotic:jota-backend-c");
        message.append('\n').append("- Graal Native Image: add dependency com.qxotic:jota-graal");
        message.append('\n')
                .append(
                        "- Explicit backend override: -Djota.native.backend=<backend-id>"
                                + " (e.g. panama, c, hip, cuda, mojo, opencl, metal)");
        message.append('\n')
                .append("- Ensure required backends are allowed by include/exclude filters");
        message.append('\n')
                .append("- Run with -Djota.runtime.probe.log=true for backend warnings");
        return new IllegalStateException(message.toString());
    }

    private static String formatDevices(Iterable<Device> devices) {
        StringJoiner joiner = new StringJoiner(", ");
        for (Device device : devices) {
            joiner.add(device.runtimeId());
        }
        String value = joiner.toString();
        return value.isEmpty() ? "<none>" : value;
    }

    private static void registerProvider(
            DefaultRuntimeRegistry registry,
            DeviceRuntimeProvider provider,
            Set<String> includedBackends,
            Set<String> excludedBackends) {
        DeviceType deviceType = provider.deviceType();
        String providerId = deviceType.id();
        RuntimeProbe filterProbe = backendFilterProbe(provider, includedBackends, excludedBackends);
        if (filterProbe != null) {
            registry.addDiagnostic(new RuntimeDiagnostic(providerId, deviceType, filterProbe));
            return;
        }
        RuntimeProbe probe;
        try {
            probe = provider.probe();
        } catch (Throwable t) {
            probe = RuntimeProbe.error("Runtime probe threw unexpectedly", t);
        }
        registry.addDiagnostic(new RuntimeDiagnostic(providerId, deviceType, probe));
        if (!probe.isAvailable()) {
            logUnavailableRuntime(providerId, deviceType, probe);
            return;
        }
        try {
            int deviceCount = provider.deviceCount();
            for (int i = 0; i < deviceCount; i++) {
                Device logical = new Device(deviceType, i);
                final int deviceIndex = i;
                registry.registerFactory(logical, () -> provider.create(deviceIndex));
            }
        } catch (Throwable t) {
            RuntimeProbe failure =
                    RuntimeProbe.error("Runtime provider failed to register factories", t);
            registry.addDiagnostic(new RuntimeDiagnostic(providerId, deviceType, failure));
            logUnavailableRuntime(providerId, deviceType, failure);
        }
    }

    private static void logUnavailableRuntime(
            String id, DeviceType deviceType, RuntimeProbe probe) {
        if (!LOG_RUNTIME_WARNINGS) {
            return;
        }
        StringBuilder warning = new StringBuilder();
        warning.append("[jota-runtime] WARNING: backend '")
                .append(id)
                .append("' on device ")
                .append(deviceType.id())
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
        private static final Environment INSTANCE;

        static {
            RuntimeRegistry runtimes = buildDefaultRuntimes();
            Device nativeDevice = runtimes.resolve("native");
            INSTANCE = new Environment(nativeDevice, DataTypeImpl.defaultFloatValue(), runtimes);
        }
    }
}
