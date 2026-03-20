package com.qxotic.jota.runtime.metal;

import com.qxotic.jota.DeviceType;
import com.qxotic.jota.runtime.DeviceRuntime;
import com.qxotic.jota.runtime.spi.DeviceRuntimeProvider;
import com.qxotic.jota.runtime.spi.RuntimeProbe;
import java.io.IOException;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.TimeUnit;

public final class MetalRuntimeProvider extends DeviceRuntimeProvider {

    @Override
    public DeviceType deviceType() {
        return DeviceType.METAL;
    }

    @Override
    public RuntimeProbe probe() {
        if (!isMacOs()) {
            return RuntimeProbe.unsupportedHardware(
                    "Metal runtime is only supported on macOS",
                    "Run on macOS with an Apple Metal-capable GPU");
        }
        if (!isCommandAvailable(List.of("xcrun", "--version"))) {
            return RuntimeProbe.missingSoftware(
                    "Xcode command line tools executable is not available: xcrun",
                    "Install Xcode command line tools and ensure xcrun is on PATH");
        }
        if (!isCommandAvailable(List.of("xcrun", "-sdk", "macosx", "-find", "metal"))) {
            return RuntimeProbe.missingSoftware(
                    "Metal compiler tool is not available",
                    "Install Xcode command line tools that provide the 'metal' compiler");
        }
        if (!isCommandAvailable(List.of("xcrun", "-sdk", "macosx", "-find", "metallib"))) {
            return RuntimeProbe.missingSoftware(
                    "metallib tool is not available",
                    "Install Xcode command line tools that provide the 'metallib' tool");
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
    public DeviceRuntime create(int deviceIndex) {
        return new MetalDeviceRuntime(deviceType().deviceIndex(deviceIndex));
    }

    private static boolean isMacOs() {
        String os = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
        return os.contains("mac");
    }

    private static boolean isCommandAvailable(List<String> command) {
        Process process = null;
        try {
            process = new ProcessBuilder(command).start();
            if (!process.waitFor(3, TimeUnit.SECONDS)) {
                process.destroyForcibly();
                process.waitFor(1, TimeUnit.SECONDS);
                return false;
            }
            return process.exitValue() == 0;
        } catch (IOException | InterruptedException e) {
            if (e instanceof InterruptedException) {
                Thread.currentThread().interrupt();
            }
            return false;
        } finally {
            if (process != null) {
                process.destroy();
            }
        }
    }
}
