package com.qxotic.jota.runtime.panama;

import com.qxotic.jota.Device;
import com.qxotic.jota.DeviceType;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.runtime.ComputeEngine;
import com.qxotic.jota.runtime.DeviceRuntime;
import com.qxotic.jota.runtime.DiskKernelCache;
import com.qxotic.jota.runtime.FileKernelProgramStore;
import com.qxotic.jota.runtime.KernelBackend;
import com.qxotic.jota.runtime.KernelCachePaths;
import com.qxotic.jota.runtime.KernelProgramStore;
import com.qxotic.jota.runtime.KernelService;
import com.qxotic.jota.runtime.nativeimpl.NativeMemoryFactory;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;

public final class PanamaDeviceRuntime implements DeviceRuntime {

    private final MemoryDomain<MemorySegment> memoryDomain;
    private final ComputeEngine computeEngine;
    private final KernelService kernelService;

    public PanamaDeviceRuntime() {
        this(NativeMemoryFactory.createDomain(), DiskKernelCache.defaultCache(DeviceType.PANAMA));
    }

    public PanamaDeviceRuntime(MemoryDomain<MemorySegment> memoryDomain, DiskKernelCache cache) {
        this.memoryDomain = Objects.requireNonNull(memoryDomain, "memoryDomain");
        this.computeEngine = new PanamaLirComputeEngine(memoryDomain, cache);
        KernelBackend backend = new JavaKernelBackend(memoryDomain, cache);
        Path programRoot = KernelCachePaths.programRoot(DeviceType.PANAMA);
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
                "JVM (" + System.getProperty("java.vm.name") + ")",
                "device.vendor",
                System.getProperty("java.vm.vendor"),
                "device.architecture",
                System.getProperty("os.arch"),
                "device.driver.version",
                System.getProperty("java.runtime.version"),
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
