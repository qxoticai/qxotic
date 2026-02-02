package ai.qxotic.jota.tensor;

import ai.qxotic.jota.Device;
import ai.qxotic.jota.Environment;
import ai.qxotic.jota.ir.tir.IotaConstant;
import ai.qxotic.jota.ir.tir.TIRGraph;
import ai.qxotic.jota.ir.tir.TIRInterpreter;
import ai.qxotic.jota.ir.tir.TIRNode;
import ai.qxotic.jota.memory.MemoryContext;
import ai.qxotic.jota.memory.MemoryView;
import ai.qxotic.jota.panama.PanamaLIRKernelExecutor;
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
        Device device = inputTensors.get(0).device();
        MemoryContext<?> context = Environment.current().backend(device).memoryContext();

        if (device == Device.PANAMA && requiresKernel(graph)) {
            @SuppressWarnings("unchecked")
            MemoryContext<java.lang.foreign.MemorySegment> panamaContext =
                    (MemoryContext<java.lang.foreign.MemorySegment>) context;
            List<Tensor> lirInputs = collectLirInputs(graph, inputTensors);
            return new PanamaLIRKernelExecutor(DiskKernelCache.defaultCache())
                    .execute(graph, lirInputs, panamaContext);
        }

        @SuppressWarnings({"unchecked", "rawtypes"})
        List<MemoryView<?>> inputs =
                (List<MemoryView<?>>)
                        (List<?>) inputTensors.stream().map(Tensor::materialize).toList();

        List<?> outputs = TIRInterpreter.execute(graph, inputs, context);

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

    TIRGraph graph() {
        return graph;
    }

    List<Tensor> inputTensors() {
        return inputTensors;
    }
}
