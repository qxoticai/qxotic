package ai.qxotic.jota.panama;

import ai.qxotic.jota.Device;
import ai.qxotic.jota.ir.tir.TIRGraph;
import ai.qxotic.jota.memory.MemoryContext;
import ai.qxotic.jota.memory.MemoryView;
import ai.qxotic.jota.tensor.ComputeBackend;
import ai.qxotic.jota.tensor.DiskKernelCache;
import ai.qxotic.jota.tensor.Tensor;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Objects;

final class PanamaLirComputeBackend implements ComputeBackend {

    private final MemoryContext<MemorySegment> context;
    private final PanamaLIRKernelExecutor executor;

    PanamaLirComputeBackend(MemoryContext<MemorySegment> context, DiskKernelCache cache) {
        this.context = Objects.requireNonNull(context, "context");
        this.executor = new PanamaLIRKernelExecutor(Objects.requireNonNull(cache, "cache"));
    }

    @Override
    public Device device() {
        return context.device();
    }

    @Override
    public MemoryView<?> execute(TIRGraph graph, List<Tensor> inputs) {
        return executor.execute(graph, inputs, context);
    }
}
