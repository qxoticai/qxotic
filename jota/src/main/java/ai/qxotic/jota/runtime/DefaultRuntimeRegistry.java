package ai.qxotic.jota.runtime;

import ai.qxotic.jota.Device;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

public final class DefaultRuntimeRegistry implements RuntimeRegistry {

    private final Map<Device, DeviceRuntime> runtimes = new ConcurrentHashMap<>();
    private volatile DeviceRuntime nativeDeviceRuntime;

    public static DefaultRuntimeRegistry withNative(DeviceRuntime deviceRuntime) {
        DefaultRuntimeRegistry registry = new DefaultRuntimeRegistry();
        registry.registerNative(deviceRuntime);
        return registry;
    }

    @Override
    public void register(DeviceRuntime deviceRuntime) {
        Objects.requireNonNull(deviceRuntime, "deviceRuntime");
        runtimes.put(deviceRuntime.device(), deviceRuntime);
        if (nativeDeviceRuntime == null && deviceRuntime.device().equals(Device.PANAMA)) {
            nativeDeviceRuntime = deviceRuntime;
        }
    }

    public void registerNative(DeviceRuntime deviceRuntime) {
        register(deviceRuntime);
        nativeDeviceRuntime = deviceRuntime;
    }

    @Override
    public DeviceRuntime runtimeFor(Device device) {
        DeviceRuntime deviceRuntime = runtimes.get(device);
        if (deviceRuntime == null) {
            throw new IllegalStateException("No backend registered for " + device);
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
            throw new IllegalStateException("No native backend registered");
        }
        return deviceRuntime;
    }

    @Override
    public Set<Device> devices() {
        return Set.copyOf(runtimes.keySet());
    }
}
