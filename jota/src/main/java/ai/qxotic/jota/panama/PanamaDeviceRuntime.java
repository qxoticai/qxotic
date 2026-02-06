package ai.qxotic.jota.panama;

import ai.qxotic.jota.Device;
import ai.qxotic.jota.backend.DeviceRuntime;
import ai.qxotic.jota.backend.FileKernelProgramStore;
import ai.qxotic.jota.backend.KernelProgramStore;
import ai.qxotic.jota.backend.KernelService;
import ai.qxotic.jota.memory.MemoryDomain;
import ai.qxotic.jota.tensor.ComputeEngine;
import ai.qxotic.jota.tensor.DiskKernelCache;
import ai.qxotic.jota.tensor.KernelBackend;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.Objects;
import java.util.Optional;

public final class PanamaDeviceRuntime implements DeviceRuntime {

    private final MemoryDomain<MemorySegment> memoryDomain;
    private final ComputeEngine computeEngine;
    private final KernelService kernelService;

    public PanamaDeviceRuntime() {
        this(PanamaFactory.createDomain(), DiskKernelCache.defaultCache());
    }

    public PanamaDeviceRuntime(MemoryDomain<MemorySegment> memoryDomain, DiskKernelCache cache) {
        this.memoryDomain = Objects.requireNonNull(memoryDomain, "memoryDomain");
        this.computeEngine = new PanamaLirComputeEngine(memoryDomain, cache);
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
