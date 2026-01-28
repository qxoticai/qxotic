package ai.qxotic.jota.tensor;

import ai.qxotic.jota.memory.MemoryContext;
import java.lang.foreign.MemorySegment;

public interface JitKernel {

    void execute(MemoryContext<MemorySegment> context, KernelArgs args);
}
