package com.qxotic.jota.runtime.mojo;

import com.qxotic.jota.Device;
import com.qxotic.jota.runtime.DeviceRuntime;
import com.qxotic.jota.runtime.hip.HipRuntimeProvider;
import com.qxotic.jota.runtime.mojo.bridge.MojoRuntime;
import com.qxotic.jota.runtime.spi.DeviceRuntimeProvider;
import com.qxotic.jota.runtime.spi.RuntimeProbe;

/** Registers Mojo as an explicit backend while reusing HIP device execution in v1. */
public final class MojoRuntimeProvider implements DeviceRuntimeProvider {

    @Override
    public String id() {
        return "mojo";
    }

    @Override
    public Device device() {
        return Device.MOJO;
    }

    @Override
    public int priority() {
        return 50;
    }

    @Override
    public RuntimeProbe probe() {
        RuntimeProbe hipProbe = new HipRuntimeProvider().probe();
        if (!hipProbe.isAvailable()) {
            return RuntimeProbe.missingSoftware(
                    "Mojo backend requires HIP runtime/toolchain availability", hipProbe.hint());
        }
        if (!MojoRuntime.isAvailable()) {
            return RuntimeProbe.missingSoftware(
                    "Mojo JNI runtime library is not available",
                    "Ensure libjota_mojo is built and discoverable on java.library.path or in the"
                            + " backend JAR native resources");
        }
        return RuntimeProbe.available("Mojo backend available via HIP execution runtime");
    }

    @Override
    public DeviceRuntime create() {
        return new MojoDeviceRuntime();
    }
}
