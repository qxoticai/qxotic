package com.qxotic.jota.runtime;

import com.qxotic.jota.Device;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.function.Supplier;

public final class DefaultRuntimeRegistry implements RuntimeRegistry {

    /** name/alias → resolved concrete Device (e.g., "native" → Device(PANAMA, 0)) */
    private final Map<String, Device> aliases = new ConcurrentHashMap<>();

    /** concrete device → lazy factory */
    private final Map<Device, Supplier<DeviceRuntime>> factories = new ConcurrentHashMap<>();

    /** cache of created runtimes (populated lazily) */
    private final Map<Device, DeviceRuntime> runtimes = new ConcurrentHashMap<>();

    private final CopyOnWriteArrayList<RuntimeDiagnostic> diagnostics =
            new CopyOnWriteArrayList<>();

    /**
     * Registers a lazy factory for a concrete device.
     *
     * @param device the concrete (indexed) device
     * @param factory supplier that creates the runtime on first access
     */
    public void registerFactory(Device device, Supplier<DeviceRuntime> factory) {
        Objects.requireNonNull(device, "device");
        Objects.requireNonNull(factory, "factory");
        Supplier<DeviceRuntime> existing = factories.putIfAbsent(device, factory);
        if (existing != null) {
            throw new IllegalStateException("Factory already registered for device " + device);
        }
        // Auto-register the runtimeId as an alias → this concrete device (first one wins)
        aliases.putIfAbsent(device.runtimeId(), device);
    }

    /**
     * Registers an alias mapping.
     *
     * @param aliasId the alias name (e.g., "native", "default")
     * @param target the concrete device to resolve to
     */
    public void registerAlias(String aliasId, Device target) {
        Objects.requireNonNull(aliasId, "aliasId");
        Objects.requireNonNull(target, "target");
        aliases.put(aliasId.strip().toLowerCase(), target);
    }

    @Override
    public Device resolve(String nameOrAlias) {
        Objects.requireNonNull(nameOrAlias, "nameOrAlias");
        String key = nameOrAlias.strip().toLowerCase();
        Device resolved = aliases.get(key);
        if (resolved == null) {
            throw new IllegalStateException("No device registered for alias '" + nameOrAlias + "'");
        }
        return resolved;
    }

    @Override
    public DeviceRuntime runtimeFor(Device device) {
        Objects.requireNonNull(device, "device");
        return runtimes.computeIfAbsent(
                device,
                d -> {
                    Supplier<DeviceRuntime> factory = factories.get(d);
                    if (factory == null) {
                        throw new IllegalStateException("No runtime registered for " + d);
                    }
                    DeviceRuntime runtime = factory.get();
                    if (runtime == null) {
                        throw new IllegalStateException(
                                "Runtime factory returned null for device " + d);
                    }
                    Device runtimeDevice = runtime.device();
                    if (!d.equals(runtimeDevice)) {
                        throw new IllegalStateException(
                                "Runtime factory for device "
                                        + d
                                        + " returned runtime bound to "
                                        + runtimeDevice);
                    }
                    return runtime;
                });
    }

    @Override
    public DeviceRuntime runtimeFor(String nameOrAlias) {
        return runtimeFor(resolve(nameOrAlias));
    }

    @Override
    public boolean hasRuntime(Device device) {
        return factories.containsKey(device);
    }

    @Override
    public boolean hasRuntime(String nameOrAlias) {
        try {
            Device concrete = resolve(nameOrAlias);
            return factories.containsKey(concrete);
        } catch (IllegalStateException e) {
            return false;
        }
    }

    @Override
    public Set<Device> devices() {
        return Set.copyOf(factories.keySet());
    }

    @Override
    public List<RuntimeDiagnostic> diagnostics() {
        return List.copyOf(diagnostics);
    }

    public void addDiagnostic(RuntimeDiagnostic diagnostic) {
        diagnostics.add(Objects.requireNonNull(diagnostic, "diagnostic"));
    }
}
