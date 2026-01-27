package ai.qxotic.jota.panama;

import ai.qxotic.jota.Device;
import ai.qxotic.jota.backend.Backend;
import ai.qxotic.jota.backend.FileKernelProgramStore;
import ai.qxotic.jota.backend.KernelPipeline;
import ai.qxotic.jota.backend.KernelProgramStore;
import ai.qxotic.jota.memory.MemoryContext;
import ai.qxotic.jota.tensor.ComputeEngine;
import ai.qxotic.jota.tensor.DiskKernelCache;
import ai.qxotic.jota.tensor.KernelBackend;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.Objects;

public final class PanamaBackend implements Backend {

    private final MemoryContext<MemorySegment> context;
    private final ComputeEngine computeEngine;
    private final KernelPipeline kernelPipeline;

    public PanamaBackend() {
        this(PanamaFactory.context(), DiskKernelCache.defaultCache());
    }

    public PanamaBackend(MemoryContext<MemorySegment> context, DiskKernelCache cache) {
        this.context = Objects.requireNonNull(context, "context");
        this.computeEngine = new JavaComputeEngine(context, cache);
        KernelBackend backend = new JavaKernelBackend(context, cache);
        KernelProgramStore store =
                new FileKernelProgramStore(
                        Path.of("__kernels").resolve(Device.PANAMA.leafName()).resolve("programs"));
        this.kernelPipeline = new KernelPipeline(backend, null, null, store);
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
    public KernelPipeline kernelPipeline() {
        return kernelPipeline;
    }
}
