package ai.qxotic.jota.tensor;

import ai.qxotic.jota.Device;
import ai.qxotic.jota.Layout;
import ai.qxotic.jota.Shape;

public interface ExecutionContext {
    Device device();

    Tensor allocateOutput(Shape shape, ai.qxotic.jota.DataType dtype);

    Tensor allocateOutput(Shape shape, ai.qxotic.jota.DataType dtype, Layout layout);

    ScratchAllocator scratch();

    void barrier();
}
