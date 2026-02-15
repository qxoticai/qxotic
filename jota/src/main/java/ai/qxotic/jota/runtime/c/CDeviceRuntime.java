package ai.qxotic.jota.runtime.c;

import ai.qxotic.jota.Device;
import ai.qxotic.jota.memory.MemoryDomain;
import ai.qxotic.jota.runtime.DeviceRuntime;
import ai.qxotic.jota.runtime.FileKernelProgramStore;
import ai.qxotic.jota.runtime.KernelProgramStore;
import ai.qxotic.jota.runtime.KernelService;
import ai.qxotic.jota.tensor.ComputeEngine;
import ai.qxotic.jota.tensor.KernelBackend;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.Objects;
import java.util.Optional;

public final class CDeviceRuntime implements DeviceRuntime {

    private final CMemoryDomain memoryDomain;
    private final ComputeEngine computeEngine;
    private final KernelService kernelService;

    public CDeviceRuntime() {
        this(new CMemoryDomain());
    }

    public CDeviceRuntime(CMemoryDomain memoryDomain) {
        this.memoryDomain = Objects.requireNonNull(memoryDomain, "memoryDomain");
        this.computeEngine = new CComputeEngine(memoryDomain);
        KernelBackend backend = new CKernelBackend();
        Path programRoot = Path.of("__kernels").resolve(Device.C.leafName()).resolve("programs");
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
