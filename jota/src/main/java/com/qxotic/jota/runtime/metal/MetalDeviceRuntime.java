package com.qxotic.jota.runtime.metal;

import com.qxotic.jota.Device;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.runtime.ComputeEngine;
import com.qxotic.jota.runtime.DeviceRuntime;
import com.qxotic.jota.runtime.EagerKernelFactory;
import com.qxotic.jota.runtime.FileKernelProgramStore;
import com.qxotic.jota.runtime.KernelBackend;
import com.qxotic.jota.runtime.KernelProgramStore;
import com.qxotic.jota.runtime.KernelService;
import java.nio.file.Path;
import java.util.Objects;
import java.util.Optional;

public final class MetalDeviceRuntime implements DeviceRuntime {

    private final MetalMemoryDomain memoryDomain;
    private final ComputeEngine computeEngine;
    private final KernelService kernelService;
    private final EagerKernels eagerKernels;

    public MetalDeviceRuntime() {
        this(MetalMemoryDomain.instance());
    }

    public MetalDeviceRuntime(MetalMemoryDomain memoryDomain) {
        this.memoryDomain = Objects.requireNonNull(memoryDomain, "memoryDomain");
        this.computeEngine = new MetalComputeEngine(memoryDomain.device());
        KernelBackend backend = new MetalKernelBackend();
        Path programRoot =
                Path.of("__kernels").resolve(Device.METAL.leafName()).resolve("programs");
        KernelProgramStore sourceStore = new FileKernelProgramStore(programRoot.resolve("source"));
        KernelProgramStore binaryStore = new FileKernelProgramStore(programRoot.resolve("binary"));
        this.kernelService = new KernelService(backend, sourceStore, binaryStore);
        this.eagerKernels = EagerKernelFactory.create(this);
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
    public Optional<EagerKernels> eagerKernels() {
        return Optional.of(eagerKernels);
    }
}
