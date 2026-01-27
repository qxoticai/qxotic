package ai.qxotic.jota.hip;

import ai.qxotic.jota.Device;
import ai.qxotic.jota.backend.Backend;
import ai.qxotic.jota.backend.FileKernelProgramStore;
import ai.qxotic.jota.backend.KernelPipeline;
import ai.qxotic.jota.backend.KernelProgramStore;
import ai.qxotic.jota.memory.MemoryContext;
import ai.qxotic.jota.tensor.ComputeEngine;
import ai.qxotic.jota.tensor.KernelArgsBuilder;
import ai.qxotic.jota.tensor.KernelBackend;
import ai.qxotic.jota.tensor.KernelHarness;
import ai.qxotic.jota.tensor.KernelProgramGenerator;
import java.nio.file.Path;
import java.util.Objects;

public final class HipBackend implements Backend {

    private final HipMemoryContext context;
    private final ComputeEngine computeEngine;
    private final KernelPipeline kernelPipeline;

    public HipBackend() {
        this(HipMemoryContext.instance());
    }

    public HipBackend(HipMemoryContext context) {
        this.context = Objects.requireNonNull(context, "context");
        this.computeEngine = new HipComputeEngine(context);
        KernelBackend backend = new HipKernelBackend();
        KernelProgramGenerator generator = new HipKernelProgramGenerator();
        KernelHarness harness = new KernelHarness(generator, backend, new KernelArgsBuilder());
        KernelProgramStore store =
                new FileKernelProgramStore(
                        Path.of("__kernels").resolve(Device.HIP.leafName()).resolve("programs"));
        this.kernelPipeline = new KernelPipeline(backend, generator, harness, store);
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
