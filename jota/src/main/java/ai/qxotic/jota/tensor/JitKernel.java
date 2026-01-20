package ai.qxotic.jota.tensor;

import ai.qxotic.jota.memory.MemoryContext;
import ai.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;

public interface JitKernel {

    void execute(
            MemoryContext<?> context,
            MemoryView<MemorySegment>[] inputs,
            MemoryView<MemorySegment> output);
}
