package com.qxotic.jota.tensor;

import com.qxotic.jota.ir.tir.TIRGraph;
import java.util.Map;
import java.util.Optional;

public final class TensorTracing {

    private TensorTracing() {}

    public static Optional<TIRGraph> tracedGraph(Tensor tensor) {
        return InternalTensorAccess.computation(tensor)
                .filter(IRComputation.class::isInstance)
                .map(IRComputation.class::cast)
                .map(IRComputation::graph);
    }

    public static Optional<Map<String, Object>> computationAttributes(Tensor tensor) {
        return InternalTensorAccess.computation(tensor).map(LazyComputation::attributes);
    }
}
