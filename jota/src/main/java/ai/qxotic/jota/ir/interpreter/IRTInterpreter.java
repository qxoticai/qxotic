package ai.qxotic.jota.ir.interpreter;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Layout;
import ai.qxotic.jota.ir.irt.IRGraph;
import ai.qxotic.jota.ir.irt.IRTNode;
import ai.qxotic.jota.memory.Memory;
import ai.qxotic.jota.memory.MemoryAccess;
import ai.qxotic.jota.memory.MemoryContext;
import ai.qxotic.jota.memory.MemoryView;
import ai.qxotic.jota.panama.PanamaFactory;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;

public final class IRTInterpreter {

    private IRTInterpreter() {}

    public static List<MemoryView<MemorySegment>> execute(
            IRGraph graph, List<MemoryView<?>> inputs, MemoryContext<?> context) {

        try (IRTEvalContext evalContext = IRTEvalContext.create(inputs, context)) {
            List<MemoryView<MemorySegment>> arenaOutputs = new ArrayList<>();

            for (IRTNode outputNode : graph.outputs()) {
                MemoryView<MemorySegment> output = evalContext.evaluate(outputNode);
                arenaOutputs.add(output);
            }

            List<MemoryView<MemorySegment>> persistentOutputs = new ArrayList<>();
            MemoryAccess<MemorySegment> memAccess =
                    (MemoryAccess<MemorySegment>) context.memoryAccess();

            for (MemoryView<MemorySegment> arenaOutput : arenaOutputs) {
                DataType dtype = arenaOutput.dataType();
                Layout layout = arenaOutput.layout();
                long size = layout.shape().size();

                Memory<MemorySegment> persistentMemory =
                        PanamaFactory.onHeapAllocator().allocateMemory(dtype, size);
                MemoryView<MemorySegment> persistentOutput =
                        MemoryView.of(persistentMemory, 0, dtype, layout);

                for (long i = 0; i < size; i++) {
                    long offset = ai.qxotic.jota.Indexing.linearToOffset(arenaOutput, i);
                    long persistentOffset =
                            ai.qxotic.jota.Indexing.linearToOffset(persistentOutput, i);

                    if (dtype == DataType.FP32) {
                        float value = memAccess.readFloat(arenaOutput.memory(), offset);
                        memAccess.writeFloat(persistentOutput.memory(), persistentOffset, value);
                    } else if (dtype == DataType.FP64) {
                        double value = memAccess.readDouble(arenaOutput.memory(), offset);
                        memAccess.writeDouble(persistentOutput.memory(), persistentOffset, value);
                    } else if (dtype == DataType.I32) {
                        int value = memAccess.readInt(arenaOutput.memory(), offset);
                        memAccess.writeInt(persistentOutput.memory(), persistentOffset, value);
                    } else if (dtype == DataType.I64) {
                        long value = memAccess.readLong(arenaOutput.memory(), offset);
                        memAccess.writeLong(persistentOutput.memory(), persistentOffset, value);
                    } else {
                        throw new UnsupportedOperationException(
                                "Output copy not implemented for dtype: " + dtype);
                    }
                }

                persistentOutputs.add(persistentOutput);
            }

            return persistentOutputs;
        }
    }
}
