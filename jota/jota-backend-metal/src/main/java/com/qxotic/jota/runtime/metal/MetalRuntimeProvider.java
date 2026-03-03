package com.qxotic.jota.runtime.metal;

import com.qxotic.jota.Device;
import com.qxotic.jota.runtime.DeviceRuntime;
import com.qxotic.jota.runtime.spi.DeviceRuntimeProvider;
import com.qxotic.jota.runtime.spi.RuntimeProbe;
import java.util.Locale;

public final class MetalRuntimeProvider implements DeviceRuntimeProvider {

    @Override
    public String id() {
        return "metal";
    }

    @Override
    public Device device() {
        return Device.METAL;
    }

    @Override
    public RuntimeProbe probe() {
        if (!isMacOs()) {
            return RuntimeProbe.unsupportedHardware(
                    "Metal runtime is only supported on macOS",
                    "Run on macOS with an Apple Metal-capable GPU");
        }
        if (!MetalRuntime.isAvailable()) {
            return RuntimeProbe.missingSoftware(
                    "Metal JNI runtime library is not available",
                    "Ensure libjota_metal is installed and discoverable");
        }
        try {
            int count = MetalRuntime.deviceCount();
            if (count <= 0) {
                return RuntimeProbe.unsupportedHardware(
                        "Metal runtime is present but no compatible GPU was detected",
                        "Ensure Metal is available and at least one MTL device is visible");
            }
            return RuntimeProbe.available("Metal runtime available with " + count + " device(s)");
        } catch (RuntimeException e) {
            return RuntimeProbe.misconfigured(
                    "Metal runtime is loaded but device probing failed",
                    "Check macOS permissions and Metal runtime configuration",
                    e);
        }
    }

    @Override
    public DeviceRuntime create() {
        return new MetalDeviceRuntime();
    }

    private static boolean isMacOs() {
        String os = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
        return os.contains("mac");
    }
}
