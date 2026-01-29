package ai.qxotic.jota.tensor;

import ai.qxotic.jota.ir.irt.IRGraph;
import ai.qxotic.jota.ir.irt.IRTNode;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

/**
 * Orchestrates IR-T tracing and creates IRComputation. Similar to Tracer but uses IR-T instead of
 * ExprNode.
 */
public final class IRTracer {

    private IRTracer() {}

    /** Traces a single-input function to IR-T. */
    public static Tensor trace(Tensor input, Function<Tensor, Tensor> fn) {
        return trace(List.of(input), tensors -> fn.apply(tensors.get(0)));
    }

    /** Traces a multi-input function to IR-T. */
    public static Tensor trace(List<Tensor> inputs, Function<List<Tensor>, Tensor> fn) {
        TraceInputs traceInputs = traceInputs(inputs);
        Tensor output =
                TensorOpsContext.with(
                        new IRTensorOps(), () -> fn.apply(new ArrayList<>(traceInputs.tensors())));
        IRTensor irtOutput = (IRTensor) output;
        IRGraph graph = new IRGraph(traceInputs.nodes(), List.of(irtOutput.node()));
        return Tensor.lazy(
                new IRComputation(graph, inputs),
                output.dataType(),
                output.layout(),
                output.device());
    }

    private static TraceInputs traceInputs(List<Tensor> inputs) {
        List<IRTNode> nodes = new ArrayList<>();
        List<Tensor> tensors = new ArrayList<>();
        Map<Integer, Tensor> tensorMap = new HashMap<>();

        for (int i = 0; i < inputs.size(); i++) {
            Tensor input = inputs.get(i);
            IRTNode node;

            if (input.isMaterialized() || input.computation().isEmpty()) {
                node = new ai.qxotic.jota.ir.irt.TensorInput(i, input.dataType(), input.layout());
            } else {
                LazyComputation comp = input.computation().orElseThrow();
                if (comp instanceof IRComputation irComp) {
                    node = remapInputs(irComp.graph(), i, irComp.inputTensors());
                } else {
                    node =
                            new ai.qxotic.jota.ir.irt.TensorInput(
                                    i, input.dataType(), input.layout());
                }
            }

            nodes.add(node);
            tensors.add(new IRTensor(node, input.device()));
            tensorMap.put(i, input);
        }

        return new TraceInputs(nodes, tensors, tensorMap);
    }

    private static IRTNode remapInputs(IRGraph graph, int baseIndex, List<Tensor> originalInputs) {
        IRTNode oldRoot = graph.outputs().get(0);
        return remapNode(oldRoot, baseIndex);
    }

    private static IRTNode remapNode(IRTNode node, int indexOffset) {
        return node.accept(new IRTNodeRemapper(indexOffset));
    }

    private record TraceInputs(
            List<IRTNode> nodes, List<Tensor> tensors, Map<Integer, Tensor> tensorMap) {
        public List<IRTNode> nodes() {
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
