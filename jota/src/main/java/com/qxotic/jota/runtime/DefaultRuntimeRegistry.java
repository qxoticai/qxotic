package com.qxotic.jota.runtime;

import com.qxotic.jota.Device;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;

public final class DefaultRuntimeRegistry implements RuntimeRegistry {

    private final Map<Device, DeviceRuntime> runtimes = new ConcurrentHashMap<>();
    private final CopyOnWriteArrayList<RuntimeDiagnostic> diagnostics =
            new CopyOnWriteArrayList<>();
    private volatile DeviceRuntime nativeDeviceRuntime;

    public static DefaultRuntimeRegistry withNative(DeviceRuntime deviceRuntime) {
        DefaultRuntimeRegistry registry = new DefaultRuntimeRegistry();
        registry.registerNative(deviceRuntime);
        return registry;
    }

    @Override
    public void register(DeviceRuntime deviceRuntime) {
        Objects.requireNonNull(deviceRuntime, "deviceRuntime");
        Device device = deviceRuntime.device();
        DeviceRuntime existing = runtimes.putIfAbsent(device, deviceRuntime);
        if (existing != null) {
            throw new IllegalStateException("Runtime already registered for device " + device);
        }
        if (nativeDeviceRuntime == null && deviceRuntime.device().equals(Device.PANAMA)) {
            nativeDeviceRuntime = deviceRuntime;
        }
    }

    public void registerNative(DeviceRuntime deviceRuntime) {
        Objects.requireNonNull(deviceRuntime, "deviceRuntime");
        if (!deviceRuntime.device().equals(Device.PANAMA)) {
            throw new IllegalArgumentException(
                    "Native runtime must target "
                            + Device.PANAMA
                            + ", got "
                            + deviceRuntime.device());
        }
        if (nativeDeviceRuntime != null
                && !nativeDeviceRuntime.device().equals(deviceRuntime.device())) {
            throw new IllegalStateException(
                    "Native runtime already configured for " + nativeDeviceRuntime.device());
        }
        register(deviceRuntime);
        nativeDeviceRuntime = deviceRuntime;
    }

    @Override
    public DeviceRuntime runtimeFor(Device device) {
        DeviceRuntime deviceRuntime = runtimes.get(device);
        if (deviceRuntime == null) {
            throw new IllegalStateException("No runtime registered for " + device);
        }
        return deviceRuntime;
    }

    @Override
    public boolean hasRuntime(Device device) {
        return runtimes.containsKey(device);
    }

    @Override
    public DeviceRuntime nativeRuntime() {
        DeviceRuntime deviceRuntime = nativeDeviceRuntime;
        if (deviceRuntime == null) {
            throw new IllegalStateException("No native runtime registered");
        }
        return deviceRuntime;
    }

    @Override
    public Set<Device> devices() {
        return Set.copyOf(runtimes.keySet());
    }

    @Override
    public List<RuntimeDiagnostic> diagnostics() {
        return List.copyOf(diagnostics);
    }

    @Override
    public List<RuntimeDiagnostic> diagnosticsFor(Device device) {
        Objects.requireNonNull(device, "device");
        return diagnostics.stream().filter(it -> it.device().equals(device)).toList();
    }

    public void addDiagnostic(RuntimeDiagnostic diagnostic) {
        diagnostics.add(Objects.requireNonNull(diagnostic, "diagnostic"));
    }
}
