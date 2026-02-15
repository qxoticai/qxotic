package ai.qxotic.jota.runtime.c;

import ai.qxotic.jota.Device;
import ai.qxotic.jota.runtime.DeviceRuntime;
import ai.qxotic.jota.runtime.spi.DeviceRuntimeProvider;
import ai.qxotic.jota.runtime.spi.RuntimeProbe;

public final class CRuntimeProvider implements DeviceRuntimeProvider {

    @Override
    public String id() {
        return "c";
    }

    @Override
    public Device device() {
        return Device.C;
    }

    @Override
    public RuntimeProbe probe() {
        if (!CNative.isAvailable()) {
            return RuntimeProbe.missingSoftware(
                    "C JNI runtime library is not available",
                    "Ensure libjota_c is on java.library.path / LD_LIBRARY_PATH");
        }
        return RuntimeProbe.available("C runtime available");
    }

    @Override
    public DeviceRuntime create() {
        return new CDeviceRuntime();
    }
}
