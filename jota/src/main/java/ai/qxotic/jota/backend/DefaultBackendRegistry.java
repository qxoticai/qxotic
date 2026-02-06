package ai.qxotic.jota.backend;

import ai.qxotic.jota.Device;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

public final class DefaultBackendRegistry implements BackendRegistry {

    private final Map<Device, DeviceRuntime> backends = new ConcurrentHashMap<>();
    private volatile DeviceRuntime nativeDeviceRuntime;

    public static DefaultBackendRegistry withNative(DeviceRuntime deviceRuntime) {
        DefaultBackendRegistry registry = new DefaultBackendRegistry();
        registry.registerNative(deviceRuntime);
        return registry;
    }

    @Override
    public void register(DeviceRuntime deviceRuntime) {
        Objects.requireNonNull(deviceRuntime, "backend");
        backends.put(deviceRuntime.device(), deviceRuntime);
        if (nativeDeviceRuntime == null && deviceRuntime.device().equals(Device.PANAMA)) {
            nativeDeviceRuntime = deviceRuntime;
        }
    }

    public void registerNative(DeviceRuntime deviceRuntime) {
        register(deviceRuntime);
        nativeDeviceRuntime = deviceRuntime;
    }

    @Override
    public DeviceRuntime backend(Device device) {
        DeviceRuntime deviceRuntime = backends.get(device);
        if (deviceRuntime == null) {
            throw new IllegalStateException("No backend registered for " + device);
        }
        return deviceRuntime;
    }

    @Override
    public boolean hasBackend(Device device) {
        return backends.containsKey(device);
    }

    @Override
    public DeviceRuntime nativeBackend() {
        DeviceRuntime deviceRuntime = nativeDeviceRuntime;
        if (deviceRuntime == null) {
            throw new IllegalStateException("No native backend registered");
        }
        return deviceRuntime;
    }

    @Override
    public Set<Device> devices() {
        return Set.copyOf(backends.keySet());
    }
}
