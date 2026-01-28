package ai.qxotic.jota.panama;

import ai.qxotic.jota.Device;
import ai.qxotic.jota.backend.Backend;
import ai.qxotic.jota.backend.FileKernelProgramStore;
import ai.qxotic.jota.backend.KernelProgramStore;
import ai.qxotic.jota.backend.KernelService;
import ai.qxotic.jota.memory.MemoryContext;
import ai.qxotic.jota.tensor.ComputeEngine;
import ai.qxotic.jota.tensor.DiskKernelCache;
import ai.qxotic.jota.tensor.KernelBackend;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.Objects;
import java.util.Optional;

public final class PanamaBackend implements Backend {

    private final MemoryContext<MemorySegment> context;
    private final ComputeEngine computeEngine;
    private final KernelService kernelService;

    public PanamaBackend() {
        this(PanamaFactory.createContext(), DiskKernelCache.defaultCache());
    }

    public PanamaBackend(MemoryContext<MemorySegment> context, DiskKernelCache cache) {
        this.context = Objects.requireNonNull(context, "context");
        this.computeEngine = new JavaComputeEngine(context, cache);
        KernelBackend backend = new JavaKernelBackend(context, cache);
        Path programRoot =
                Path.of("__kernels").resolve(Device.PANAMA.leafName()).resolve("programs");
        KernelProgramStore sourceStore = new FileKernelProgramStore(programRoot.resolve("source"));
        KernelProgramStore binaryStore = new FileKernelProgramStore(programRoot.resolve("binary"));
        this.kernelService = new KernelService(backend, null, sourceStore, binaryStore);
    }

    @Override
    public Device device() {
        return context.device();
    }

    @Override
    public MemoryContext<MemorySegment> memoryContext() {
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
