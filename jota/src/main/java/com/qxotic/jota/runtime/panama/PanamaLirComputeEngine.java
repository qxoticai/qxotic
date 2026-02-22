package com.qxotic.jota.runtime.panama;

import com.qxotic.jota.Device;
import com.qxotic.jota.ir.tir.TIRGraph;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.tensor.ComputeEngine;
import com.qxotic.jota.tensor.DiskKernelCache;
import com.qxotic.jota.tensor.Tensor;
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
