package com.qxotic.jota.runtime;

import com.qxotic.jota.Device;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.function.Function;

public final class DefaultRuntimeRegistry implements RuntimeRegistry {

    /** concrete device -> lazy factory */
    private final Map<Device, Function<Device, DeviceRuntime>> factories =
            new ConcurrentHashMap<>();

    /** cache of created runtimes (populated lazily) */
    private final Map<Device, DeviceRuntime> runtimes = new ConcurrentHashMap<>();

    private final CopyOnWriteArrayList<RuntimeDiagnostic> diagnostics =
            new CopyOnWriteArrayList<>();

    /**
     * Registers a lazy factory for a concrete device.
     *
     * @param device the concrete (indexed) device
     * @param factory function that creates the runtime for the requested device on first access
     */
    public void registerFactory(Device device, Function<Device, DeviceRuntime> factory) {
        Objects.requireNonNull(device, "device");
        Objects.requireNonNull(factory, "factory");
        Function<Device, DeviceRuntime> existing = factories.putIfAbsent(device, factory);
        if (existing != null) {
            throw new IllegalStateException("Factory already registered for device " + device);
        }
    }

    @Override
    public DeviceRuntime runtimeFor(Device device) {
        Objects.requireNonNull(device, "device");
        return runtimes.computeIfAbsent(
                device,
                d -> {
                    Function<Device, DeviceRuntime> factory = factories.get(d);
                    if (factory == null) {
                        throw new IllegalStateException("No runtime registered for " + d);
                    }
                    DeviceRuntime runtime = factory.apply(d);
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
    public boolean hasRuntimeFor(Device device) {
        return factories.containsKey(device);
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
