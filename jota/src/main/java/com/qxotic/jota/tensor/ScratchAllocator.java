package com.qxotic.jota.tensor;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import java.lang.foreign.MemorySegment;

public interface ScratchAllocator {
    Tensor allocate(Shape shape, DataType dtype);

    MemorySegment allocateBytes(long size);
}
