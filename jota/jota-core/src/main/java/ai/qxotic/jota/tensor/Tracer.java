package ai.qxotic.jota.tensor;

import ai.qxotic.jota.Device;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.ir.tir.TIRGraph;
import ai.qxotic.jota.ir.tir.TIRNode;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.BiFunction;
import java.util.function.Function;

/**
 * Orchestrates IR-T tracing and creates IRComputation. Similar to Tracer but uses IR-T instead of
 * ExprNode.
 */
public final class Tracer {

    private Tracer() {}

    /** Traces a single-input function to IR-T. */
    public static Tensor trace(Tensor input, Function<Tensor, Tensor> fn) {
        return trace(List.of(input), tensors -> fn.apply(tensors.get(0)));
    }

    /** Traces a two-input function to IR-T. */
    public static Tensor trace(Tensor left, Tensor right, BiFunction<Tensor, Tensor, Tensor> fn) {
        Objects.requireNonNull(left, "left");
        Objects.requireNonNull(right, "right");
        Objects.requireNonNull(fn, "fn");
        return trace(List.of(left, right), tensors -> fn.apply(tensors.get(0), tensors.get(1)));
    }

    /** Traces a three-input function to IR-T. */
    public static Tensor trace(
            Tensor first,
            Tensor second,
            Tensor third,
            TriFunction<Tensor, Tensor, Tensor, Tensor> fn) {
        Objects.requireNonNull(first, "first");
        Objects.requireNonNull(second, "second");
        Objects.requireNonNull(third, "third");
        Objects.requireNonNull(fn, "fn");
        return trace(
                List.of(first, second, third),
                tensors -> fn.apply(tensors.get(0), tensors.get(1), tensors.get(2)));
    }

    /** Traces a multi-input function to IR-T. */
    public static Tensor trace(List<Tensor> inputs, Function<List<Tensor>, Tensor> fn) {
        TraceInputs traceInputs = traceInputs(inputs);
        Tensor output =
                TensorOpsContext.with(
                        new IRTensorOps(), () -> fn.apply(new ArrayList<>(traceInputs.tensors())));
        IRTensor irtOutput = (IRTensor) output;
        TIRGraph graph = new TIRGraph(traceInputs.nodes(), List.of(irtOutput.node()));
        return Tensor.lazy(
                new IRComputation(graph, inputs),
                output.dataType(),
                output.layout(),
                output.device());
    }

    private static TraceInputs traceInputs(List<Tensor> inputs) {
        List<TIRNode> nodes = new ArrayList<>();
        List<Tensor> tensors = new ArrayList<>();
        Map<Integer, Tensor> tensorMap = new HashMap<>();

        for (int i = 0; i < inputs.size(); i++) {
            Tensor input = inputs.get(i);
            TIRNode node;
            boolean scalarBroadcast = input.layout().stride().isAllZeros();

            if (scalarBroadcast && input.device().root() == Device.CPU) {
                node = new ai.qxotic.jota.ir.tir.ScalarInput(i, input.dataType(), input.shape());
            } else if (input.isMaterialized() || input.computation().isEmpty()) {
                // Materialized tensor - create a TensorInput
                node = new ai.qxotic.jota.ir.tir.TensorInput(i, input.dataType(), input.layout());
            } else {
                LazyComputation comp = input.computation().orElseThrow();
                if (comp instanceof RangeComputation range) {
                    // Iota constant - computed from loop index
                    node =
                            new ai.qxotic.jota.ir.tir.IotaConstant(
                                    range.count(), input.dataType(), Shape.flat(range.count()));
                } else if (comp instanceof ConstantComputation constComp) {
                    // Constant value - inline as ScalarConstant
                    node =
                            ai.qxotic.jota.ir.tir.ScalarConstant.broadcast(
                                    constComp.rawBits(),
                                    constComp.dataType(),
                                    input.layout().shape());
                } else {
                    // IRComputation or other - treat as buffer input
                    // The computation will be executed first, and its result used as input
                    node =
                            new ai.qxotic.jota.ir.tir.TensorInput(
                                    i, input.dataType(), input.layout());
                }
            }

            nodes.add(node);
            tensors.add(new IRTensor(node, input.device()));
            tensorMap.put(i, input);
        }

        return new TraceInputs(nodes, tensors, tensorMap);
    }

    private record TraceInputs(
            List<TIRNode> nodes, List<Tensor> tensors, Map<Integer, Tensor> tensorMap) {
        public List<TIRNode> nodes() {
            return nodes;
        }

        public List<Tensor> tensors() {
            return tensors;
        }

        public Map<Integer, Tensor> tensorMap() {
            return tensorMap;
        }
    }
}
