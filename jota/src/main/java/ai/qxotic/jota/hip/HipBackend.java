package ai.qxotic.jota.hip;

import ai.qxotic.jota.Device;
import ai.qxotic.jota.backend.Backend;
import ai.qxotic.jota.backend.FileKernelProgramStore;
import ai.qxotic.jota.backend.KernelProgramStore;
import ai.qxotic.jota.backend.KernelService;
import ai.qxotic.jota.memory.MemoryContext;
import ai.qxotic.jota.tensor.ComputeEngine;
import ai.qxotic.jota.tensor.KernelBackend;
import ai.qxotic.jota.tensor.KernelProgramGenerator;
import java.nio.file.Path;
import java.util.Objects;
import java.util.Optional;

public final class HipBackend implements Backend {

    private final HipMemoryContext context;
    private final ComputeEngine computeEngine;
    private final KernelService kernelService;

    public HipBackend() {
        this(HipMemoryContext.instance());
    }

    public HipBackend(HipMemoryContext context) {
        this.context = Objects.requireNonNull(context, "context");
        this.computeEngine = new HipComputeEngine(context);
        KernelBackend backend = new HipKernelBackend();
        KernelProgramGenerator generator = new HipKernelProgramGenerator();
        Path programRoot = Path.of("__kernels").resolve(Device.HIP.leafName()).resolve("programs");
        KernelProgramStore sourceStore = new FileKernelProgramStore(programRoot.resolve("source"));
        KernelProgramStore binaryStore = new FileKernelProgramStore(programRoot.resolve("binary"));
        this.kernelService = new KernelService(backend, generator, sourceStore, binaryStore);
    }

    @Override
    public Device device() {
        return context.device();
    }

    @Override
    public MemoryContext<?> memoryContext() {
        return context;
    }

    @Override
    public ComputeEngine computeEngine() {
        return computeEngine;
    }

    @Override
    public Optional<KernelService> kernels() {
        return Optional.of(kernelService);
    }
}
