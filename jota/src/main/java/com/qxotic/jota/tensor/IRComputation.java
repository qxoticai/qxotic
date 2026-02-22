package com.qxotic.jota.tensor;

import com.qxotic.jota.Device;
import com.qxotic.jota.Environment;
import com.qxotic.jota.ir.tir.ScheduledProgram;
import com.qxotic.jota.ir.tir.TIRCSEPass;
import com.qxotic.jota.ir.tir.TIRConstantFoldingPass;
import com.qxotic.jota.ir.tir.TIRGraph;
import com.qxotic.jota.ir.tir.TIRSchedulePass;
import com.qxotic.jota.memory.MemoryView;
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
        ComputeEngine computeEngine = Environment.current().computeEngineFor(device);

        // Optimize the graph before execution
        TIRGraph optimizedGraph = optimizeGraph(graph);
        ScheduledProgram schedule = new TIRSchedulePass().run(optimizedGraph);

        if (!schedule.steps().isEmpty() && computeEngine == null) {
            throw new IllegalStateException(
                    "No compute engine available for scheduled lazy execution on device "
                            + device.name());
        }

        return new ScheduledExecutor().execute(schedule, computeEngine, inputTensors);
    }

    /**
     * Optimizes the TIR graph by running CSE and constant folding passes. This reduces graph size
     * by eliminating redundant subexpressions.
     */
    TIRGraph optimizeGraph(TIRGraph inputGraph) {
        TIRGraph result = inputGraph;

        // Run CSE pass to eliminate common subexpressions
        result = new TIRCSEPass().run(result);

        // Run constant folding to simplify constant expressions
        result = new TIRConstantFoldingPass().run(result);

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
