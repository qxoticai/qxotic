package com.qxotic.jota.runtime.cuda;

import com.qxotic.jota.DeviceType;
import com.qxotic.jota.runtime.DeviceRuntime;
import com.qxotic.jota.runtime.spi.DeviceRuntimeProvider;
import com.qxotic.jota.runtime.spi.RuntimeProbe;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.TimeUnit;

public final class CudaRuntimeProvider extends DeviceRuntimeProvider {

    @Override
    public DeviceType deviceType() {
        return DeviceType.CUDA;
    }

    @Override
    public RuntimeProbe probe() {
        String os = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
        if (os.contains("mac") || os.contains("darwin")) {
            return RuntimeProbe.unsupportedHardware(
                    "CUDA backend is unsupported on macOS",
                    "Use panama, opencl, metal, or c backend on this platform");
        }
        if (!CudaRuntime.isAvailable()) {
            String loadError = CudaRuntime.loadError();
            String message = "CUDA JNI runtime library is not available";
            if (loadError != null && !loadError.isBlank()) {
                message = message + ": " + loadError;
            }
            return RuntimeProbe.missingSoftware(
                    message,
                    "Ensure libjota_cuda and CUDA runtime libraries are installed and"
                            + " discoverable");
        }
        if (!isNvccAvailable()) {
            String nvcc = nvccExecutable();
            return RuntimeProbe.missingSoftware(
                    "CUDA toolchain executable is not available: " + nvcc,
                    "Install NVIDIA CUDA nvcc and ensure it is on PATH, or set "
                            + CudaKernelBackend.NVCC_PROPERTY
                            + " / "
                            + CudaKernelBackend.NVCC_ENV
                            + " to a valid nvcc binary");
        }
        try {
            int count = CudaRuntime.deviceCount();
            if (count <= 0) {
                return RuntimeProbe.unsupportedHardware(
                        "CUDA runtime is present but no compatible GPU was detected",
                        "Install supported NVIDIA GPU drivers/CUDA and verify CUDA runtime can see"
                                + " a device");
            }
            return RuntimeProbe.available("CUDA runtime available with " + count + " device(s)");
        } catch (UnsupportedOperationException e) {
            return RuntimeProbe.misconfigured(
                    "CUDA runtime is loaded but device probing is not supported",
                    "Rebuild native CUDA JNI with full CUDA runtime support",
                    e);
        } catch (RuntimeException e) {
            return RuntimeProbe.misconfigured(
                    "CUDA runtime is loaded but device probing failed",
                    "Check CUDA installation, permissions, and runtime environment variables",
                    e);
        }
    }

    @Override
    public DeviceRuntime create(long deviceIndex) {
        return new CudaDeviceRuntime(deviceType().deviceIndex(deviceIndex));
    }

    @Override
    public Map<String, String> properties(int deviceIndex) {
        if (!CudaRuntime.isAvailable()) {
            return Map.of();
        }
        var props = new LinkedHashMap<String, String>();
        props.put("device.name", CudaRuntime.deviceName(deviceIndex));
        props.put("device.vendor", "NVIDIA");
        props.put("device.architecture", CudaRuntime.deviceArchName(deviceIndex));
        props.put("device.kind", "gpu");
        return Map.copyOf(props);
    }

    @Override
    public Set<String> capabilities(int deviceIndex) {
        if (!CudaRuntime.isAvailable()) {
            return Set.of();
        }
        return Set.of(
                "gpu", "fp32", "fp64", "int8", "kernel.compilation", "atomic.32", "atomic.64");
    }

    private static String nvccExecutable() {
        return CudaKernelBackend.resolveNvccExecutable();
    }

    private static boolean isNvccAvailable() {
        String nvcc = nvccExecutable();
        Process process = null;
        try {
            List<String> command = new ArrayList<>();
            command.add(nvcc);
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
