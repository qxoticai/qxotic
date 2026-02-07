package ai.qxotic.jota.runtime.spi;

import ai.qxotic.jota.Device;
import ai.qxotic.jota.runtime.DeviceRuntime;

public interface DeviceRuntimeProvider {

    String id();

    Device device();

    default int priority() {
        return 0;
    }

    RuntimeProbe probe();

    DeviceRuntime create();
}
