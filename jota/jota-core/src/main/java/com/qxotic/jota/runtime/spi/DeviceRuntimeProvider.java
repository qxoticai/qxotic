package com.qxotic.jota.runtime.spi;

import com.qxotic.jota.DeviceType;
import com.qxotic.jota.runtime.DeviceRuntime;

public interface DeviceRuntimeProvider {

    /** The device type this provider serves (e.g., DeviceType.CUDA). */
    DeviceType deviceType();

    default int priority() {
        return 0;
    }

    RuntimeProbe probe();

    /** Number of available device indices for this runtime. */
    default int deviceCount() {
        return 1;
    }

    /** Create a runtime for the given device index. */
    DeviceRuntime create(int deviceIndex);
}
