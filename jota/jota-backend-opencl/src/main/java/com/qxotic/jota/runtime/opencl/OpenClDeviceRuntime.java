package com.qxotic.jota.runtime.opencl;

import com.qxotic.jota.Device;
import com.qxotic.jota.DeviceType;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.runtime.ComputeEngine;
import com.qxotic.jota.runtime.DeviceRuntime;
import com.qxotic.jota.runtime.FileKernelProgramStore;
import com.qxotic.jota.runtime.KernelBackend;
import com.qxotic.jota.runtime.KernelCachePaths;
import com.qxotic.jota.runtime.KernelProgramStore;
import com.qxotic.jota.runtime.KernelService;
import java.nio.file.Path;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;

public final class OpenClDeviceRuntime implements DeviceRuntime {

    private final OpenClMemoryDomain memoryDomain;
    private final ComputeEngine computeEngine;
    private final KernelService kernelService;

    public OpenClDeviceRuntime() {
        this(DeviceType.OPENCL.deviceIndex(selectedDeviceIndex()));
    }

    public OpenClDeviceRuntime(Device device) {
        this(new OpenClMemoryDomain(device));
    }

    public OpenClDeviceRuntime(OpenClMemoryDomain memoryDomain) {
        var probe = new OpenClRuntimeProvider().probe();
        if (!probe.isAvailable()) {
            String hint = probe.hint() == null ? "" : " Hint: " + probe.hint();
            throw new IllegalStateException(
                    "OpenCL backend unavailable ["
                            + probe.status()
                            + "]: "
                            + probe.message()
                            + hint);
        }
        this.memoryDomain = Objects.requireNonNull(memoryDomain, "memoryDomain");
        this.computeEngine = new OpenClComputeEngine(memoryDomain.device());
        KernelBackend backend = new OpenClKernelBackend();
        Path programRoot = KernelCachePaths.programRoot(DeviceType.OPENCL);
        KernelProgramStore sourceStore = new FileKernelProgramStore(programRoot.resolve("source"));
        KernelProgramStore binaryStore = new FileKernelProgramStore(programRoot.resolve("binary"));
        this.kernelService = new KernelService(backend, sourceStore, binaryStore);
    }

    @Override
    public Device device() {
        return memoryDomain.device();
    }

    @Override
    public MemoryDomain<?> memoryDomain() {
        return memoryDomain;
    }

    @Override
    public ComputeEngine computeEngine() {
        return computeEngine;
    }

    @Override
    public Optional<KernelService> kernelService() {
        return Optional.of(kernelService);
    }

    @Override
    public Map<String, String> properties() {
        return Map.of(
                "device.name",
                OpenClRuntime.deviceName(),
                "device.vendor",
                OpenClRuntime.deviceVendor(),
                "device.kind",
                OpenClRuntime.selectedDeviceType().toLowerCase(),
                "opencl.platform",
                OpenClRuntime.selectedPlatformName());
    }

    @Override
    public Set<String> capabilities() {
        return Set.of(
                OpenClRuntime.selectedDeviceType().toLowerCase(), "fp32", "kernel.compilation");
    }

    @Override
    public String toString() {
        return "DeviceRuntime{device=" + device() + "}";
    }

    private static int selectedDeviceIndex() {
        String token = System.getProperty(OpenClRuntime.DEVICE_INDEX_PROPERTY, "0");
        try {
            int index = Integer.parseInt(token.trim());
            return Math.max(index, 0);
        } catch (NumberFormatException ignored) {
            return 0;
        }
    }
}
