package com.qxotic.jota.tensor;

import com.qxotic.jota.Device;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;

public interface ExecutionContext {
    Device device();

    Tensor allocateOutput(Shape shape, com.qxotic.jota.DataType dtype);

    Tensor allocateOutput(Shape shape, com.qxotic.jota.DataType dtype, Layout layout);

    ScratchAllocator scratch();

    void barrier();
}
