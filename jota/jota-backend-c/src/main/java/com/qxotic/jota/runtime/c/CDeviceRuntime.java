package com.qxotic.jota.runtime.c;

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
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;

public final class CDeviceRuntime implements DeviceRuntime {

    private final CMemoryDomain memoryDomain;
    private final ComputeEngine computeEngine;
    private final KernelService kernelService;

    public CDeviceRuntime() {
        this(DeviceType.C.deviceIndex(0));
    }

    public CDeviceRuntime(Device device) {
        this(new CMemoryDomain(device));
    }

    public CDeviceRuntime(CMemoryDomain memoryDomain) {
        this.memoryDomain = Objects.requireNonNull(memoryDomain, "memoryDomain");
        this.computeEngine = new CComputeEngine(memoryDomain);
        KernelBackend backend = new CKernelBackend();
        Path programRoot = KernelCachePaths.programRoot(DeviceType.C);
        KernelProgramStore sourceStore = new FileKernelProgramStore(programRoot.resolve("source"));
        KernelProgramStore binaryStore = new FileKernelProgramStore(programRoot.resolve("binary"));
        this.kernelService = new KernelService(backend, sourceStore, binaryStore);
    }

    @Override
    public Device device() {
        return memoryDomain.device();
    }

    @Override
    public MemoryDomain<MemorySegment> memoryDomain() {
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
    public boolean supportsNativeRuntimeAlias() {
        return true;
    }

    @Override
    public Map<String, String> properties() {
        Runtime rt = Runtime.getRuntime();
        return Map.of(
                "device.name",
                "C Host",
                "device.vendor",
                System.getProperty("os.name"),
                "device.architecture",
                System.getProperty("os.arch"),
                "memory.global.bytes",
                Long.toString(rt.maxMemory()),
                "compute.units",
                Integer.toString(rt.availableProcessors()),
                "device.kind",
                "cpu");
    }

    @Override
    public Set<String> capabilities() {
        return Set.of(
                "cpu", "fp32", "fp64", "kernel.compilation", "native.runtime", "unified.memory");
    }

    @Override
    public String toString() {
        return "DeviceRuntime{device=" + device() + "}";
    }
}
