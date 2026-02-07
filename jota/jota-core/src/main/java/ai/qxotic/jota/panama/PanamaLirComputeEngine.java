package ai.qxotic.jota.panama;

import ai.qxotic.jota.Device;
import ai.qxotic.jota.ir.tir.TIRGraph;
import ai.qxotic.jota.memory.MemoryDomain;
import ai.qxotic.jota.memory.MemoryView;
import ai.qxotic.jota.tensor.ComputeEngine;
import ai.qxotic.jota.tensor.DiskKernelCache;
import ai.qxotic.jota.tensor.Tensor;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Objects;

final class PanamaLirComputeEngine implements ComputeEngine {

    private final MemoryDomain<MemorySegment> memoryDomain;
    private final PanamaLIRKernelExecutor executor;

    PanamaLirComputeEngine(MemoryDomain<MemorySegment> memoryDomain, DiskKernelCache cache) {
        this.memoryDomain = Objects.requireNonNull(memoryDomain, "memoryDomain");
        this.executor = new PanamaLIRKernelExecutor(Objects.requireNonNull(cache, "cache"));
    }

    @Override
    public Device device() {
        return memoryDomain.device();
    }

    @Override
    public MemoryView<?> execute(TIRGraph graph, List<Tensor> inputs) {
        return executor.execute(graph, inputs, memoryDomain);
    }
}
