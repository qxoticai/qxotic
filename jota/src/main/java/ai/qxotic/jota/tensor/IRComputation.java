package ai.qxotic.jota.tensor;

import ai.qxotic.jota.Device;
import ai.qxotic.jota.Environment;
import ai.qxotic.jota.backend.DeviceRuntime;
import ai.qxotic.jota.ir.tir.IotaConstant;
import ai.qxotic.jota.ir.tir.TIRGraph;
import ai.qxotic.jota.ir.tir.TIRInterpreter;
import ai.qxotic.jota.ir.tir.TIRNode;
import ai.qxotic.jota.memory.MemoryDomain;
import ai.qxotic.jota.memory.MemoryView;
import java.util.List;
import java.util.Map;
import java.util.Objects;

final class IRComputation implements LazyComputation {

    private final TIRGraph graph;
    private final List<Tensor> inputTensors;

    IRComputation(TIRGraph graph, List<Tensor> inputTensors) {
        this.graph = Objects.requireNonNull(graph);
        this.inputTensors = List.copyOf(Objects.requireNonNull(inputTensors));
    }

    @Override
    public Op operation() {
        return new Op() {
            @Override
            public String name() {
                return "ir-graph";
            }

            @Override
            public OpKind kind() {
                return OpKind.SPECIAL;
            }
        };
    }

    @Override
    public List<Tensor> inputs() {
        return inputTensors;
    }

    @Override
    public Map<String, Object> attributes() {
        return Map.of("graph", graph);
    }

    @Override
    public MemoryView<?> execute() {
        Device device =
                inputTensors.isEmpty() ? Device.defaultDevice() : inputTensors.get(0).device();
        DeviceRuntime deviceRuntime = Environment.current().backend(device);
        MemoryDomain<?> domain = deviceRuntime.memoryDomain();
        ComputeEngine computeEngine = Environment.current().computeBackendFor(device);

        // Optimize the graph before execution
        TIRGraph optimizedGraph = optimizeGraph(graph);

        if (requiresKernel(optimizedGraph)) {
            List<Tensor> lirInputs = collectLirInputs(optimizedGraph, inputTensors);
            if (computeEngine == null) {
                throw new UnsupportedOperationException(
                        "LIR execution required for device " + device + " but backend is missing");
            }
            return computeEngine.execute(optimizedGraph, lirInputs);
        }

        @SuppressWarnings({"unchecked", "rawtypes"})
        List<MemoryView<?>> inputs =
                (List<MemoryView<?>>)
                        (List<?>) inputTensors.stream().map(Tensor::materialize).toList();

        List<?> outputs = TIRInterpreter.execute(optimizedGraph, inputs, domain);

        @SuppressWarnings("unchecked")
        MemoryView<?> output = (MemoryView<?>) outputs.get(0);

        return output;
    }

    private static boolean requiresKernel(TIRGraph graph) {
        for (TIRNode output : graph.outputs()) {
            if (containsComputeNode(output)) {
                return true;
            }
        }
        return false;
    }

    private static boolean containsComputeNode(TIRNode node) {
        return switch (node) {
            case ai.qxotic.jota.ir.tir.UnaryOp __ -> true;
            case ai.qxotic.jota.ir.tir.BinaryOp __ -> true;
            case ai.qxotic.jota.ir.tir.TernaryOp __ -> true;
            case ai.qxotic.jota.ir.tir.ReductionOp __ -> true;
            case ai.qxotic.jota.ir.tir.CastOp __ -> true;
            case ai.qxotic.jota.ir.tir.ViewTransform vt -> containsComputeNode(vt.input());
            case ai.qxotic.jota.ir.tir.Contiguous contig -> containsComputeNode(contig.input());
            default -> false;
        };
    }

    private static List<Tensor> collectLirInputs(TIRGraph graph, List<Tensor> inputs) {
        List<Tensor> lirInputs = new java.util.ArrayList<>();
        for (int i = 0; i < graph.inputs().size(); i++) {
            if (graph.inputs().get(i) instanceof IotaConstant) {
                continue;
            }
            lirInputs.add(inputs.get(i));
        }
        return lirInputs;
    }

    /**
     * Optimizes the TIR graph by running CSE and constant folding passes. This reduces graph size
     * by eliminating redundant subexpressions.
     */
    TIRGraph optimizeGraph(TIRGraph inputGraph) {
        TIRGraph result = inputGraph;

        // Run CSE pass to eliminate common subexpressions
        // result = new TIRCSEPass().run(result);

        // Run constant folding to simplify constant expressions
        // result = new TIRConstantFoldingPass().run(result);

        // Validate the optimized graph
        // result = new TIRValidationPass().run(result);

        return result;
    }

    TIRGraph graph() {
        return graph;
    }

    List<Tensor> inputTensors() {
        return inputTensors;
    }
}
