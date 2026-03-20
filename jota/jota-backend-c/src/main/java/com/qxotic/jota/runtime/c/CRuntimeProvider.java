package com.qxotic.jota.runtime.c;

import com.qxotic.jota.DeviceType;
import com.qxotic.jota.runtime.DeviceRuntime;
import com.qxotic.jota.runtime.spi.DeviceRuntimeProvider;
import com.qxotic.jota.runtime.spi.RuntimeProbe;
import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.TimeUnit;

public final class CRuntimeProvider extends DeviceRuntimeProvider {

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
    public DeviceRuntime create(long deviceIndex) {
        return new CDeviceRuntime(deviceType().deviceIndex(deviceIndex));
    }

    @Override
    public Map<String, String> properties(int deviceIndex) {
        Runtime rt = Runtime.getRuntime();
        var props = new LinkedHashMap<String, String>();
        props.put("device.name", "C Host");
        props.put("device.vendor", System.getProperty("os.name"));
        props.put("device.architecture", System.getProperty("os.arch"));
        props.put("memory.global.bytes", Long.toString(rt.maxMemory()));
        props.put("compute.units", Integer.toString(rt.availableProcessors()));
        props.put("device.kind", "cpu");
        return Map.copyOf(props);
    }

    @Override
    public Set<String> capabilities(int deviceIndex) {
        return Set.of(
                "fp32", "fp64", "kernel.compilation", "native.runtime", "unified.memory", "cpu");
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
