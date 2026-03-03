package com.qxotic.jota.runtime.spi;

import com.qxotic.jota.Device;
import com.qxotic.jota.runtime.DeviceRuntime;

public interface DeviceRuntimeProvider {

    String id();

    Device device();

    default int priority() {
        return 0;
    }

    RuntimeProbe probe();

    DeviceRuntime create();
}
