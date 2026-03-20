package com.qxotic.jota.runtime.mojo;

import com.qxotic.jota.DeviceType;
import com.qxotic.jota.runtime.DeviceRuntime;
import com.qxotic.jota.runtime.hip.HipRuntimeProvider;
import com.qxotic.jota.runtime.mojo.bridge.MojoRuntime;
import com.qxotic.jota.runtime.spi.DeviceRuntimeProvider;
import com.qxotic.jota.runtime.spi.RuntimeProbe;
import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.TimeUnit;

/** Registers Mojo as an explicit backend while reusing HIP device execution in v1. */
public final class MojoRuntimeProvider extends DeviceRuntimeProvider {

    @Override
    public DeviceType deviceType() {
        return DeviceType.MOJO;
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
        String compiler = MojoKernelBackend.resolveCompilerExecutable();
        if (!isCommandAvailable(List.of(compiler, "--version"))) {
            return RuntimeProbe.missingSoftware(
                    "Mojo toolchain executable is not available: " + compiler,
                    "Install Mojo and ensure it is on PATH, or set "
                            + MojoKernelBackend.COMPILER_PROPERTY
                            + " / "
                            + MojoKernelBackend.COMPILER_ENV
                            + " to a valid mojo binary");
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
    public DeviceRuntime create(long deviceIndex) {
        return new MojoDeviceRuntime();
    }

    @Override
    public Map<String, String> properties(int deviceIndex) {
        var props = new LinkedHashMap<String, String>();
        props.put("device.name", "Mojo (via HIP)");
        props.put("device.vendor", "Modular");
        props.put("device.kind", "gpu");
        props.put("mojo.execution.backend", "hip");
        return Map.copyOf(props);
    }

    @Override
    public Set<String> capabilities(int deviceIndex) {
        return Set.of("gpu", "fp32", "kernel.compilation");
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
