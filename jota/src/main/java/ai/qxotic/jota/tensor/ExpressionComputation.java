package ai.qxotic.jota.tensor;

import ai.qxotic.jota.memory.MemoryView;
import java.util.List;
import java.util.Map;
import java.util.Objects;

final class ExpressionComputation implements LazyComputation {

    private final ExpressionGraph graph;
    private final List<Tensor> inputs;

    ExpressionComputation(ExpressionGraph graph, List<Tensor> inputs) {
        this.graph = Objects.requireNonNull(graph, "graph");
        this.inputs = List.copyOf(inputs);
    }

    ExpressionGraph graph() {
        return graph;
    }

    @Override
    public Op operation() {
        return ExpressionOp.INSTANCE;
    }

    @Override
    public List<Tensor> inputs() {
        return inputs;
    }

    @Override
    public Map<String, Object> attributes() {
        return Map.of("graph", graph);
    }

    @Override
    public MemoryView<?> execute() {
        ComputeEngine engine = ComputeEngineContext.require();
        ComputeBackend backend = engine.backendFor(graph.root().device());
        return backend.execute(graph, inputs);
    }
}
