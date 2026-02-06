package ai.qxotic.jota.backend;

import ai.qxotic.jota.Device;
import java.util.Set;

public interface BackendRegistry {
    void register(DeviceRuntime deviceRuntime);

    DeviceRuntime backend(Device device);

    boolean hasBackend(Device device);

    DeviceRuntime nativeBackend();

    Set<Device> devices();
}
