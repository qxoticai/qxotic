package ai.qxotic.jota.tensor;

import ai.qxotic.jota.Device;
import ai.qxotic.jota.Environment;
import ai.qxotic.jota.ir.interpreter.IRTInterpreter;
import ai.qxotic.jota.ir.irt.IRGraph;
import ai.qxotic.jota.memory.MemoryContext;
import ai.qxotic.jota.memory.MemoryView;
import java.util.List;
import java.util.Map;
import java.util.Objects;

final class IRComputation implements LazyComputation {

    private final IRGraph graph;
    private final List<Tensor> inputTensors;

    IRComputation(IRGraph graph, List<Tensor> inputTensors) {
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

        @SuppressWarnings({"unchecked", "rawtypes"})
        List<MemoryView<?>> inputs =
                (List<MemoryView<?>>)
                        (List<?>) inputTensors.stream().map(Tensor::materialize).toList();

        List<?> outputs = IRTInterpreter.execute(graph, inputs, context);

        @SuppressWarnings("unchecked")
        MemoryView<?> output = (MemoryView<?>) outputs.get(0);

        return output;
    }

    IRGraph graph() {
        return graph;
    }

    List<Tensor> inputTensors() {
        return inputTensors;
    }
}
