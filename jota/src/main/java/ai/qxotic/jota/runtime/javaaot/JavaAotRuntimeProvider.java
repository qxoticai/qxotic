package ai.qxotic.jota.runtime.javaaot;

import ai.qxotic.jota.Device;
import ai.qxotic.jota.runtime.DeviceRuntime;
import ai.qxotic.jota.runtime.spi.DeviceRuntimeProvider;
import ai.qxotic.jota.runtime.spi.RuntimeProbe;

public final class JavaAotRuntimeProvider implements DeviceRuntimeProvider {

    @Override
    public String id() {
        return "java-aot";
    }

    @Override
    public Device device() {
        return Device.JAVA_AOT;
    }

    @Override
    public RuntimeProbe probe() {
        return RuntimeProbe.available("Java AOT runtime available");
    }

    @Override
    public DeviceRuntime create() {
        return new JavaAotDeviceRuntime();
    }

    @Override
    public int priority() {
        return 10;
    }
}
