package ai.qxotic.jota.tensor;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Shape;
import java.lang.foreign.MemorySegment;

public interface ScratchAllocator {
    Tensor allocate(Shape shape, DataType dtype);

    MemorySegment allocateBytes(long size);
}
