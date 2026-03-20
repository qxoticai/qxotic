package com.qxotic.jota.runtime.metal;

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

public final class MetalDeviceRuntime implements DeviceRuntime {

    private final MetalMemoryDomain memoryDomain;
    private final ComputeEngine computeEngine;
    private final KernelService kernelService;

    public MetalDeviceRuntime() {
        this(DeviceType.METAL.deviceIndex(0));
    }

    public MetalDeviceRuntime(Device device) {
        this(new MetalMemoryDomain(device));
    }

    public MetalDeviceRuntime(MetalMemoryDomain memoryDomain) {
        this.memoryDomain = Objects.requireNonNull(memoryDomain, "memoryDomain");
        this.computeEngine = new MetalComputeEngine(memoryDomain.device());
        KernelBackend backend = new MetalKernelBackend();
        Path programRoot = KernelCachePaths.programRoot(DeviceType.METAL);
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
                MetalRuntime.deviceName(),
                "device.vendor",
                "Apple",
                "device.architecture",
                System.getProperty("os.arch"),
                "device.kind",
                "gpu");
    }

    @Override
    public Set<String> capabilities() {
        return Set.of(
                "gpu",
                "fp16",
                "fp32",
                "kernel.compilation",
                "unified.memory",
                "concurrent.kernels");
    }

    @Override
    public String toString() {
        return "DeviceRuntime{device=" + device() + "}";
    }
}
