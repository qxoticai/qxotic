package com.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Device;
import com.qxotic.jota.DeviceType;
import com.qxotic.jota.Environment;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryHelpers;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.testutil.ConfiguredTestDevice;
import com.qxotic.jota.testutil.RunOnAllAvailableBackends;
import com.qxotic.jota.testutil.TensorTestReads;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

@RunOnAllAvailableBackends
class ToOpTest {

    private static MemoryDomain<MemorySegment> domain;

    @BeforeAll
    static void setUpDomain() {
        domain = Environment.current().nativeMemoryDomain();
    }

    @Test
    void toRejectsNullDevice() {
        Tensor input = Tensor.iota(4, DataType.FP32);
        assertThrows(NullPointerException.class, () -> input.to(null));
    }

    @Test
    void toReturnsSameInstanceOnSameDevice() {
        MemoryView<MemorySegment> view =
                MemoryHelpers.arange(domain, DataType.FP32, 4).view(Shape.of(4));
        Tensor input = Tensor.of(view);
        assertSame(input, input.to(input.device()));
    }

    @Test
    void toThrowsInsideTracingContext() {
        MemoryView<MemorySegment> view =
                MemoryHelpers.arange(domain, DataType.FP32, 4).view(Shape.of(4));
        Tensor input = Tensor.of(view);
        assertThrows(
                UnsupportedOperationException.class,
                () -> Tracer.trace(input, t -> t.to(DeviceType.CUDA.deviceIndex(0))));
    }

    @Test
    void toOnSameDeviceDoesNotMaterializeLazyTensor() {
        Tensor lazy = Tensor.iota(128, DataType.FP32);
        assertFalse(TensorTestInternals.isMaterialized(lazy));

        Tensor same = lazy.to(ConfiguredTestDevice.resolve());

        assertSame(lazy, same);
        assertFalse(TensorTestInternals.isMaterialized(same));
    }

    @Test
    void toCopiesIotaTensorToConfiguredTargetAsContiguous() {
        Device target = assumeTransferTarget();

        Tensor src = Tensor.iota(24, DataType.I32).view(Shape.of(2, 3, 4));
        Tensor dst = src.to(target);

        assertEquals(target, dst.device());
        assertEquals(src.shape(), dst.shape());
        assertEquals(src.dataType(), dst.dataType());
        assertEquals(src.layout(), dst.layout());
        assertTensorEqualsExact(src, dst.to(Environment.current().nativeRuntime().device()));
    }

    @Test
    void toCopiesConstantTensorToConfiguredTargetAsContiguous() {
        Device target = assumeTransferTarget();

        Tensor src = Tensor.full(7, Shape.of(3, 4)).transpose(0, 1);
        Tensor dst = src.to(target);

        assertEquals(target, dst.device());
        assertEquals(src.shape(), dst.shape());
        assertEquals(src.dataType(), dst.dataType());
        assertEquals(src.layout(), dst.layout());
        assertTensorEqualsExact(src, dst.to(Environment.current().nativeRuntime().device()));
    }

    @Test
    void toCopiesTransposedStridedTensorToConfiguredTargetAsContiguous() {
        Device target = assumeTransferTarget();

        Tensor src = Tensor.iota(12, DataType.FP32).view(Shape.of(3, 4)).transpose(0, 1);
        Tensor dst = src.to(target);

        assertEquals(target, dst.device());
        assertEquals(src.shape(), dst.shape());
        assertEquals(src.dataType(), dst.dataType());
        assertEquals(src.layout(), dst.layout());
        assertTensorEqualsExact(src, dst.to(Environment.current().nativeRuntime().device()));
    }

    @Test
    void toCopiesSlicedStridedTensorWithOffsetToConfiguredTarget() {
        Device target = assumeTransferTarget();

        Tensor src =
                Tensor.iota(40, DataType.I64).view(Shape.of(5, 8)).slice(0, 1, 5).slice(1, 1, 7, 2);
        Tensor dst = src.to(target);

        assertEquals(target, dst.device());
        assertEquals(src.shape(), dst.shape());
        assertEquals(src.dataType(), dst.dataType());
        assertEquals(src.layout(), dst.layout());
        assertTensorEqualsExact(src, dst.to(Environment.current().nativeRuntime().device()));
    }

    @Test
    void toCopiesBroadcastedTensorByLogicalElementsToConfiguredTarget() {
        Device target = assumeTransferTarget();

        Tensor src = Tensor.scalar(7.0f, DataType.FP32).broadcast(Shape.of(5, 3));
        Tensor dst = src.to(target);

        assertEquals(target, dst.device());
        assertEquals(src.shape(), dst.shape());
        assertEquals(src.dataType(), dst.dataType());
        assertEquals(src.layout(), dst.layout());
        assertTensorEqualsExact(src, dst.to(Environment.current().nativeRuntime().device()));
    }

    @Test
    void toCopiesZeroSizedTensorToConfiguredTarget() {
        Device target = assumeTransferTarget();

        Tensor src = Tensor.iota(0, DataType.FP64).view(Shape.of(0, 3));
        Tensor dst = src.to(target);

        assertEquals(target, dst.device());
        assertEquals(src.shape(), dst.shape());
        assertEquals(src.dataType(), dst.dataType());
        assertEquals(src.layout(), dst.layout());
        assertEquals(0L, dst.shape().size());
        assertTensorEqualsExact(src, dst.to(Environment.current().nativeRuntime().device()));
    }

    @Test
    void toRoundTripPreservesValuesAcrossSeveralDtypes() {
        Device target = assumeTransferTarget();

        DataType[] dtypes = {
            DataType.FP32, DataType.FP64, DataType.I32, DataType.I64, DataType.I16, DataType.I8
        };
        for (DataType dtype : dtypes) {
            Tensor src =
                    Tensor.iota(18, DataType.I64).cast(dtype).view(Shape.of(3, 6)).transpose(0, 1);
            Tensor roundTrip = src.to(target).to(Environment.current().nativeRuntime().device());
            assertTensorEqualsExact(src, roundTrip);
        }
    }

    private static Device assumeTransferTarget() {
        Device target = ConfiguredTestDevice.resolve();
        assumeTrue(
                !target.belongsTo(DeviceType.PANAMA),
                "transfer tests require non-PANAMA configured target");
        assumeTrue(
                Environment.current().runtimes().hasRuntimeFor(target),
                target + " runtime not registered in current environment");
        return target;
    }

    private static void assertTensorEqualsExact(Tensor expected, Tensor actual) {
        assertEquals(expected.shape(), actual.shape());
        assertEquals(expected.dataType(), actual.dataType());
        long n = expected.shape().size();
        for (int i = 0; i < n; i++) {
            Object leftValue = TensorTestReads.readValue(expected, i, expected.dataType());
            Object rightValue = TensorTestReads.readValue(actual, i, actual.dataType());
            assertEquals(leftValue, rightValue);
        }
    }
}
