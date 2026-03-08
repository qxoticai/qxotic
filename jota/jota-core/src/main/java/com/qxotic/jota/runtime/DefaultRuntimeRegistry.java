package com.qxotic.jota.runtime;

import com.qxotic.jota.Device;
import java.util.HashSet;
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
        if (device.equals(Device.NATIVE)) {
            throw new IllegalArgumentException(
                    "Device.NATIVE is a runtime alias and cannot be registered");
        }
        DeviceRuntime existing = runtimes.putIfAbsent(device, deviceRuntime);
        if (existing != null) {
            throw new IllegalStateException("Runtime already registered for device " + device);
        }
        if (nativeDeviceRuntime == null && deviceRuntime.supportsNativeRuntimeAlias()) {
            nativeDeviceRuntime = deviceRuntime;
        }
    }

    public void registerNative(DeviceRuntime deviceRuntime) {
        Objects.requireNonNull(deviceRuntime, "deviceRuntime");
        if (!deviceRuntime.supportsNativeRuntimeAlias()) {
            throw new IllegalArgumentException(
                    "Runtime does not support Device.NATIVE alias: "
                            + deviceRuntime.getClass().getName()
                            + " (device="
                            + deviceRuntime.device()
                            + ")");
        }
        Device device = deviceRuntime.device();
        DeviceRuntime existing = runtimes.putIfAbsent(device, deviceRuntime);
        if (existing != null && existing != deviceRuntime) {
            deviceRuntime = existing;
        }
        nativeDeviceRuntime = deviceRuntime;
    }

    @Override
    public DeviceRuntime runtimeFor(Device device) {
        if (device.equals(Device.NATIVE)) {
            return nativeRuntime();
        }
        DeviceRuntime deviceRuntime = runtimes.get(device);
        if (deviceRuntime == null) {
            throw new IllegalStateException("No runtime registered for " + device);
        }
        return deviceRuntime;
    }

    @Override
    public boolean hasRuntime(Device device) {
        if (device.equals(Device.NATIVE)) {
            return nativeDeviceRuntime != null;
        }
        return runtimes.containsKey(device);
    }

    @Override
    public DeviceRuntime nativeRuntime() {
        DeviceRuntime deviceRuntime = nativeDeviceRuntime;
        if (deviceRuntime == null) {
            throw new IllegalStateException(
                    "No native runtime registered. Available runtimes: " + runtimes.keySet());
        }
        return deviceRuntime;
    }

    @Override
    public Set<Device> devices() {
        Set<Device> devices = new HashSet<>(runtimes.keySet());
        if (nativeDeviceRuntime != null) {
            devices.add(Device.NATIVE);
        }
        return Set.copyOf(devices);
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
