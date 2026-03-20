package com.qxotic.jota.runtime.spi;

import com.qxotic.jota.DeviceType;
import com.qxotic.jota.runtime.DeviceRuntime;

public abstract class DeviceRuntimeProvider {

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

    /** Create a runtime for the given device index. */
    public abstract DeviceRuntime create(int deviceIndex);
}
