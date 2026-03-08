package com.qxotic.jota.runtime.panama;

import com.qxotic.jota.Device;
import com.qxotic.jota.runtime.DeviceRuntime;
import com.qxotic.jota.runtime.nativeimpl.NativeMemoryFactory;
import com.qxotic.jota.runtime.spi.DeviceRuntimeProvider;
import com.qxotic.jota.runtime.spi.RuntimeProbe;

public final class PanamaRuntimeProvider implements DeviceRuntimeProvider {

    @Override
    public String id() {
        return "panama";
    }

    @Override
    public Device device() {
        return Device.PANAMA;
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
    public DeviceRuntime create() {
        return new PanamaDeviceRuntime();
    }
}
