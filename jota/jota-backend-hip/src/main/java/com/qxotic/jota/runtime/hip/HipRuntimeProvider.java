package com.qxotic.jota.runtime.hip;

import com.qxotic.jota.Device;
import com.qxotic.jota.runtime.DeviceRuntime;
import com.qxotic.jota.runtime.spi.DeviceRuntimeProvider;
import com.qxotic.jota.runtime.spi.RuntimeProbe;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

public final class HipRuntimeProvider implements DeviceRuntimeProvider {

    private static final String ENV_HIPCC = "HIPCC";

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
        if (!isHipccAvailable()) {
            String hipcc = hipccExecutable();
            return RuntimeProbe.missingSoftware(
                    "HIP toolchain executable is not available: " + hipcc,
                    "Install ROCm hipcc and ensure it is on PATH, or set "
                            + ENV_HIPCC
                            + " to a valid hipcc binary");
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

    private static String hipccExecutable() {
        String fromEnv = System.getenv(ENV_HIPCC);
        if (fromEnv != null && !fromEnv.isBlank()) {
            return fromEnv.trim();
        }
        return "hipcc";
    }

    private static boolean isHipccAvailable() {
        String hipcc = hipccExecutable();
        Process process = null;
        try {
            List<String> command = new ArrayList<>();
            command.add(hipcc);
            command.add("--version");
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
