package com.qxotic.jota.runtime.panama;

import com.qxotic.jota.Device;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.runtime.ComputeEngine;
import com.qxotic.jota.runtime.DeviceRuntime;
import com.qxotic.jota.runtime.DiskKernelCache;
import com.qxotic.jota.runtime.FileKernelProgramStore;
import com.qxotic.jota.runtime.KernelBackend;
import com.qxotic.jota.runtime.KernelLaunchContext;
import com.qxotic.jota.runtime.KernelProgramStore;
import com.qxotic.jota.runtime.KernelService;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.Objects;
import java.util.Optional;

public final class PanamaDeviceRuntime implements DeviceRuntime {

    private final MemoryDomain<MemorySegment> memoryDomain;
    private final ComputeEngine computeEngine;
    private final KernelService kernelService;
    private final KernelLaunchContext launchContext;

    public PanamaDeviceRuntime() {
        this(
                PanamaFactory.createDomain(),
                DiskKernelCache.defaultCache(),
                KernelLaunchContext.disabled());
    }

    public PanamaDeviceRuntime(MemoryDomain<MemorySegment> memoryDomain, DiskKernelCache cache) {
        this(memoryDomain, cache, KernelLaunchContext.disabled());
    }

    public PanamaDeviceRuntime(
            MemoryDomain<MemorySegment> memoryDomain,
            DiskKernelCache cache,
            KernelLaunchContext launchContext) {
        this.memoryDomain = Objects.requireNonNull(memoryDomain, "memoryDomain");
        this.launchContext = Objects.requireNonNull(launchContext, "launchContext");
        this.computeEngine = new PanamaLirComputeEngine(memoryDomain, cache, this.launchContext);
        KernelBackend backend = new JavaKernelBackend(memoryDomain, cache);
        Path programRoot =
                Path.of("__kernels").resolve(Device.PANAMA.leafName()).resolve("programs");
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
}
