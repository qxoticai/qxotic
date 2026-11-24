package com.qxotic.jota.tensor;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Environment;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;
import com.qxotic.jota.ir.tir.TIRGraph;
import com.qxotic.jota.memory.MemoryView;
import java.util.List;

/** Test-only bridges into jota's package-private tracing and lazy-execution machinery. */
public final class TracedGraphs {

    private TracedGraphs() {}

    /** The TIR graph recorded by {@link Tracer} for a traced tensor. */
    public static TIRGraph graphOf(Tensor traced) {
        LazyComputation computation =
                InternalTensorAccess.computation(traced)
                        .orElseThrow(() -> new IllegalArgumentException("not a traced tensor"));
        if (computation instanceof IRComputation ir) {
            return ir.graph();
        }
        throw new IllegalArgumentException("not an IR computation: " + computation);
    }

    /**
     * Runs {@code graph} (single output) through jota's real execution path: a lazy IR computation
     * scheduled, lowered and compiled by the active backend.
     */
    public static MemoryView<?> execute(
            TIRGraph graph, List<Tensor> inputs, DataType dataType, Shape shape) {
        Tensor lazy =
                TensorFactory.lazy(
                        new IRComputation(graph, inputs),
                        dataType,
                        Layout.rowMajor(shape),
                        inputs.isEmpty()
                                ? Environment.current().defaultDevice()
                                : inputs.getFirst().device());
        return lazy.materialize();
    }
}
