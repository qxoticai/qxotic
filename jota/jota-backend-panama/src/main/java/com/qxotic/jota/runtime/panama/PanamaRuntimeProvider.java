package com.qxotic.jota.runtime.panama;

import com.qxotic.jota.DeviceType;
import com.qxotic.jota.runtime.DeviceRuntime;
import com.qxotic.jota.runtime.nativeimpl.NativeMemoryFactory;
import com.qxotic.jota.runtime.spi.DeviceRuntimeProvider;
import com.qxotic.jota.runtime.spi.RuntimeProbe;

public final class PanamaRuntimeProvider extends DeviceRuntimeProvider {

    @Override
    public DeviceType deviceType() {
        return DeviceType.PANAMA;
    }

    @Override
    public int priority() {
        return 1000;
    }

    @Override
    public RuntimeProbe probe() {
        try {
            NativeMemoryFactory.createDomain();
            return RuntimeProbe.available("Panama runtime available");
        } catch (Throwable t) {
            return RuntimeProbe.error("Panama runtime failed to initialize", t);
        }
    }

    @Override
    public DeviceRuntime create(int deviceIndex) {
        return new PanamaDeviceRuntime();
    }

    @Override
    public String toString() {
        return "PanamaRuntimeProvider[deviceType=panama, priority=" + priority() + "]";
    }
}
