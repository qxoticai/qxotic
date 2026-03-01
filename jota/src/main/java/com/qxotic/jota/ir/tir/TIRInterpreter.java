package com.qxotic.jota.ir.tir;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Indexing;
import com.qxotic.jota.Layout;
import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryAccess;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.runtime.panama.PanamaFactory;
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
            MemoryAccess<MemorySegment> memAccess =
                    (MemoryAccess<MemorySegment>) memoryDomain.directAccess();

            for (MemoryView<MemorySegment> arenaOutput : arenaOutputs) {
                DataType dtype = arenaOutput.dataType();
                Layout layout = arenaOutput.layout();
                long size = layout.shape().size();

                Memory<MemorySegment> persistentMemory =
                        PanamaFactory.onHeapAllocator().allocateMemory(dtype, size);
                MemoryView<MemorySegment> persistentOutput =
                        MemoryView.of(persistentMemory, 0, dtype, layout);

                for (long i = 0; i < size; i++) {
                    long offset = Indexing.linearToOffset(arenaOutput, i);
                    long persistentOffset = Indexing.linearToOffset(persistentOutput, i);

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
                    } else if (dtype == DataType.FP16 || dtype == DataType.BF16) {
                        short value = memAccess.readShort(arenaOutput.memory(), offset);
                        memAccess.writeShort(persistentOutput.memory(), persistentOffset, value);
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
