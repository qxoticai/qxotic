package ai.qxotic.jota.runtime.hip;

import ai.qxotic.jota.Device;
import ai.qxotic.jota.runtime.DeviceRuntime;
import ai.qxotic.jota.runtime.spi.DeviceRuntimeProvider;
import ai.qxotic.jota.runtime.spi.RuntimeProbe;

public final class HipRuntimeProvider implements DeviceRuntimeProvider {

    @Override
    public String id() {
        return "hip";
    }

    @Override
    public Device device() {
        return Device.HIP;
    }

    @Override
    public RuntimeProbe probe() {
        if (!HipRuntime.isAvailable()) {
            return RuntimeProbe.missingSoftware(
                    "HIP JNI runtime library is not available",
                    "Ensure libjota_hip and ROCm runtime libraries are installed and discoverable");
        }
        try {
            int count = HipRuntime.deviceCount();
            if (count <= 0) {
                return RuntimeProbe.unsupportedHardware(
                        "HIP runtime is present but no compatible GPU was detected",
                        "Install supported AMD GPU drivers/ROCm and verify hip runtime can see a device");
            }
            return RuntimeProbe.available("HIP runtime available with " + count + " device(s)");
        } catch (UnsupportedOperationException e) {
            return RuntimeProbe.misconfigured(
                    "HIP runtime is loaded but device probing is not supported",
                    "Rebuild native HIP JNI with full HIP runtime support",
                    e);
        } catch (RuntimeException e) {
            return RuntimeProbe.misconfigured(
                    "HIP runtime is loaded but device probing failed",
                    "Check ROCm installation, permissions, and runtime environment variables",
                    e);
        }
    }

    @Override
    public DeviceRuntime create() {
        return new HipDeviceRuntime();
    }
}
