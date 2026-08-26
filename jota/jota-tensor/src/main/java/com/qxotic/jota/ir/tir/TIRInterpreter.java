package com.qxotic.jota.ir.tir;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Indexing;
import com.qxotic.jota.Layout;
import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryAllocators;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;

public final class TIRInterpreter {

    private TIRInterpreter() {}

    public static List<MemoryView<MemorySegment>> execute(
            TIRGraph graph, List<MemoryView<?>> inputs, MemoryDomain<?> memoryDomain) {

        try (TIREvalContext evalContext = TIREvalContext.create(inputs, memoryDomain)) {
            List<MemoryView<MemorySegment>> arenaOutputs = new ArrayList<>();

            for (TIRNode outputNode : graph.outputs()) {
                MemoryView<MemorySegment> output = evalContext.evaluate(outputNode);
                arenaOutputs.add(output);
            }

            List<MemoryView<MemorySegment>> persistentOutputs = new ArrayList<>();

            for (MemoryView<MemorySegment> arenaOutput : arenaOutputs) {
                DataType dtype = arenaOutput.dataType();
                Layout layout = Layout.rowMajor(arenaOutput.shape());
                long size = layout.shape().size();

                Memory<MemorySegment> persistentMemory =
                        MemoryAllocators.ofArena(Arena.ofAuto()).allocateMemory(dtype, size);
                MemoryView<MemorySegment> persistentOutput =
                        MemoryView.of(persistentMemory, 0, dtype, layout);

                for (long i = 0; i < size; i++) {
                    long offset = Indexing.linearToOffset(arenaOutput, i);
                    long persistentOffset = Indexing.linearToOffset(persistentOutput, i);
                    MemorySegment.copy(
                            arenaOutput.memory().base(),
                            offset,
                            persistentOutput.memory().base(),
                            persistentOffset,
                            dtype.byteSize());
                }

                persistentOutputs.add(persistentOutput);
            }

            return persistentOutputs;
        }
    }
}
