package com.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Device;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryHelpers;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.memory.impl.DomainFactory;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

class ToOpsTest {

    private static MemoryDomain<MemorySegment> domain;

    @BeforeAll
    static void setUpDomain() {
        domain = DomainFactory.ofMemorySegment();
    }

    @Test
    void toReturnsSameTensorForSameDevice() {
        MemoryView<MemorySegment> view =
                MemoryHelpers.arange(domain, DataType.FP32, 4).view(Shape.of(4));
        Tensor input = Tensor.of(view);
        Tensor result =
                TensorOpsContext.with(new EagerTensorOps(domain), () -> input.to(Device.PANAMA));
        assertSame(input, result);
    }

    @Test
    void toReturnsLazyTensorForDifferentDevice() {
        MemoryView<MemorySegment> view =
                MemoryHelpers.arange(domain, DataType.FP32, 4).view(Shape.of(4));
        Tensor input = Tensor.of(view);
        assertThrows(
                UnsupportedOperationException.class,
                () -> Tracer.trace(input, t -> t.to(Device.GPU)));
    }

    @Test
    void toThrowsWhenTargetDeviceMissing() {
        MemoryView<MemorySegment> view =
                MemoryHelpers.arange(domain, DataType.FP32, 4).view(Shape.of(4));
        Tensor input = Tensor.of(view);
        assertThrows(
                IllegalStateException.class,
                () ->
                        TensorOpsContext.with(
                                new EagerTensorOps(domain), () -> input.to(Device.GPU)));
    }
}
