package com.qxotic.jota.runtime.c;

import com.qxotic.jota.DeviceType;
import com.qxotic.jota.runtime.DeviceRuntime;
import com.qxotic.jota.runtime.spi.DeviceRuntimeProvider;
import com.qxotic.jota.runtime.spi.RuntimeProbe;
import java.io.IOException;
import java.util.List;
import java.util.concurrent.TimeUnit;

public final class CRuntimeProvider implements DeviceRuntimeProvider {

    @Override
    public DeviceType deviceType() {
        return DeviceType.C;
    }

    @Override
    public RuntimeProbe probe() {
        if (!CNative.isAvailable()) {
            return RuntimeProbe.missingSoftware(
                    "C JNI runtime library is not available",
                    "Ensure libjota_c is on java.library.path / LD_LIBRARY_PATH");
        }
        String compiler = CKernelCompiler.resolveCompilerExecutable();
        if (!isCommandAvailable(List.of(compiler, "--version"))) {
            return RuntimeProbe.missingSoftware(
                    "C toolchain executable is not available: " + compiler,
                    "Install a C compiler and ensure it is on PATH, or set jota.c.compiler / CC");
        }
        return RuntimeProbe.available("C runtime available");
    }

    @Override
    public DeviceRuntime create(int deviceIndex) {
        return new CDeviceRuntime();
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
