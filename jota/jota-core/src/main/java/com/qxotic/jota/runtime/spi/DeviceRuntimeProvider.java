package com.qxotic.jota.runtime.spi;

import com.qxotic.jota.Device;
import com.qxotic.jota.DeviceType;
import com.qxotic.jota.runtime.AvailableDevice;
import com.qxotic.jota.runtime.DeviceRuntime;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.ServiceLoader;
import java.util.Set;

public abstract class DeviceRuntimeProvider {

    public static List<DeviceRuntimeProvider> availableProviders() {
        ServiceLoader<DeviceRuntimeProvider> loader =
                ServiceLoader.load(DeviceRuntimeProvider.class);
        return loader.stream()
                .map(ServiceLoader.Provider::get)
                .sorted(Comparator.comparingInt(DeviceRuntimeProvider::priority).reversed())
                .toList();
    }

    protected static DeviceType deviceType(String id) {
        return DeviceType.fromId(id);
    }

    /** The device type this provider serves (e.g., DeviceType.CUDA). */
    public abstract DeviceType deviceType();

    public int priority() {
        return 0;
    }

    public abstract RuntimeProbe probe();

    /** Number of available device indices for this runtime. */
    public int deviceCount() {
        return 1;
    }

    /** Lightweight metadata available before runtime initialization. */
    public Map<String, String> properties(int deviceIndex) {
        return Map.of();
    }

    /** Lightweight capability tags available before runtime initialization. */
    public Set<String> capabilities(int deviceIndex) {
        return Set.of();
    }

    /** Discover devices and metadata without creating runtimes. */
    public List<AvailableDevice> availableDevices() {
        RuntimeProbe probe = probe();
        if (!probe.isAvailable()) {
            return List.of();
        }
        int count = deviceCount();
        var devices = new ArrayList<AvailableDevice>(Math.max(count, 0));
        for (int i = 0; i < count; i++) {
            devices.add(
                    new AvailableDevice(
                            deviceType().deviceIndex(i), properties(i), capabilities(i)));
        }
        return List.copyOf(devices);
    }

    /**
     * Create a runtime for a concrete device. Default implementation validates runtime id matches
     * this provider and delegates to {@link #create(long)}.
     */
    public DeviceRuntime create(Device device) {
        Objects.requireNonNull(device, "device");
        DeviceType type = deviceType();
        if (!device.belongsTo(type)) {
            throw new IllegalArgumentException(
                    "Provider '"
                            + type.id()
                            + "' cannot create runtime for device '"
                            + device
                            + "'");
        }
        DeviceRuntime runtime = create(device.index());
        if (runtime == null) {
            throw new IllegalStateException("Provider '" + type.id() + "' returned null runtime");
        }
        if (!runtime.device().runtimeId().equals(device.runtimeId())) {
            throw new IllegalStateException(
                    "Provider '"
                            + type.id()
                            + "' returned runtime bound to '"
                            + runtime.device().runtimeId()
                            + "' for requested device '"
                            + device
                            + "'");
        }
        return runtime;
    }

    /** Create a runtime for the given device index. */
    public abstract DeviceRuntime create(long deviceIndex);

    @Override
    public String toString() {
        return "DeviceRuntimeProvider{deviceType="
                + deviceType().id()
                + ", priority="
                + priority()
                + "}";
    }
}
