package ai.qxotic.jota.runtime.hip;

import ai.qxotic.jota.Device;
import ai.qxotic.jota.memory.MemoryDomain;
import ai.qxotic.jota.runtime.DeviceRuntime;
import ai.qxotic.jota.runtime.FileKernelProgramStore;
import ai.qxotic.jota.runtime.KernelProgramStore;
import ai.qxotic.jota.runtime.KernelService;
import ai.qxotic.jota.tensor.ComputeEngine;
import ai.qxotic.jota.tensor.KernelBackend;
import java.nio.file.Path;
import java.util.Objects;
import java.util.Optional;

public final class HipDeviceRuntime implements DeviceRuntime {

    private final HipMemoryDomain memoryDomain;
    private final ComputeEngine computeEngine;
    private final KernelService kernelService;

    public HipDeviceRuntime() {
        this(HipMemoryDomain.instance());
    }

    public HipDeviceRuntime(HipMemoryDomain memoryDomain) {
        this.memoryDomain = Objects.requireNonNull(memoryDomain, "memoryDomain");
        this.computeEngine = new HipComputeEngine(memoryDomain.device());
        KernelBackend backend = new HipKernelBackend();
        Path programRoot = Path.of("__kernels").resolve(Device.HIP.leafName()).resolve("programs");
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
}
