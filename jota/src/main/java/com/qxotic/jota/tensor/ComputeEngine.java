package com.qxotic.jota.tensor;

import com.qxotic.jota.Device;
import com.qxotic.jota.ir.tir.TIRGraph;
import com.qxotic.jota.memory.MemoryView;
import java.util.List;

public interface ComputeEngine {

    Device device();

    MemoryView<?> execute(TIRGraph graph, List<Tensor> inputs);

    default boolean supportsParallelPrecompile() {
        return false;
    }

    default void precompile(TIRGraph graph) {
        // Default no-op.
    }
}
