package ai.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Device;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.memory.MemoryContext;
import ai.qxotic.jota.memory.MemoryHelpers;
import ai.qxotic.jota.memory.MemoryView;
import ai.qxotic.jota.memory.impl.ContextFactory;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

class ToOpsTest {

    private static MemoryContext<MemorySegment> context;

    @BeforeAll
    static void setUpContext() {
        context = ContextFactory.ofMemorySegment();
    }

    @Test
    void toReturnsSameTensorForSameDevice() {
        MemoryView<MemorySegment> view =
                MemoryHelpers.arange(context, DataType.FP32, 4).view(Shape.of(4));
        Tensor input = Tensor.of(view);
        Tensor result =
                TensorOpsContext.with(new EagerTensorOps(context), () -> input.to(Device.PANAMA));
        assertSame(input, result);
    }

    @Test
    void toReturnsLazyTensorForDifferentDevice() {
        MemoryView<MemorySegment> view =
                MemoryHelpers.arange(context, DataType.FP32, 4).view(Shape.of(4));
        Tensor input = Tensor.of(view);
        Tensor result = Tracer.trace(input, t -> t.to(Device.GPU));
        assertTrue(result.isLazy());
        assertEquals(Device.GPU, result.device());
    }

    @Test
    void toThrowsWhenTargetDeviceMissing() {
        MemoryView<MemorySegment> view =
                MemoryHelpers.arange(context, DataType.FP32, 4).view(Shape.of(4));
        Tensor input = Tensor.of(view);
        assertThrows(
                IllegalStateException.class,
                () ->
                        TensorOpsContext.with(
                                new EagerTensorOps(context), () -> input.to(Device.GPU)));
    }
}
