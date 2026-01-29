package ai.qxotic.jota.ir.interpreter;

import ai.qxotic.jota.ir.irt.IRGraph;
import ai.qxotic.jota.ir.irt.IRTNode;
import ai.qxotic.jota.memory.MemoryContext;
import ai.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;

public final class IRTInterpreter {

    private IRTInterpreter() {}

    public static List<MemoryView<MemorySegment>> execute(
            IRGraph graph, List<MemoryView<?>> inputs, MemoryContext<?> context) {

        try (IRTEvalContext evalContext = IRTEvalContext.create(inputs, context)) {
            List<MemoryView<MemorySegment>> outputs = new ArrayList<>();

            for (IRTNode outputNode : graph.outputs()) {
                MemoryView<MemorySegment> output = evalContext.evaluate(outputNode);
                outputs.add(output);
            }

            return outputs;
        }
    }
}
