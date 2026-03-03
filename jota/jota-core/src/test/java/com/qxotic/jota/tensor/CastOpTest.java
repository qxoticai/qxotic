package com.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import com.qxotic.jota.testutil.RunOnAllAvailableBackends;
import com.qxotic.jota.testutil.TensorTestReads;
import org.junit.jupiter.api.Test;

@RunOnAllAvailableBackends
class CastOpTest {

    @Test
    void castsBoolToIntegers() {
        Shape shape = Shape.of(2, 2);
        Tensor input = Tensor.of(new boolean[] {true, false, true, false}, shape);

        Tensor castI32 = Tracer.trace(input, t -> t.cast(DataType.I32));
        assertEquals(shape, castI32.materialize().shape());
        assertEquals(1, (int) TensorTestReads.readValue(castI32, 0, DataType.I32));
        assertEquals(0, (int) TensorTestReads.readValue(castI32, 1, DataType.I32));
        assertEquals(1, (int) TensorTestReads.readValue(castI32, 2, DataType.I32));
        assertEquals(0, (int) TensorTestReads.readValue(castI32, 3, DataType.I32));

        Tensor castI64 = Tracer.trace(input, t -> t.cast(DataType.I64));
        assertEquals(shape, castI64.materialize().shape());
        assertEquals(1L, (long) TensorTestReads.readValue(castI64, 0, DataType.I64));
        assertEquals(0L, (long) TensorTestReads.readValue(castI64, 1, DataType.I64));
        assertEquals(1L, (long) TensorTestReads.readValue(castI64, 2, DataType.I64));
        assertEquals(0L, (long) TensorTestReads.readValue(castI64, 3, DataType.I64));
    }

    @Test
    void castsIntToBool() {
        Shape shape = Shape.of(2, 2);
        Tensor input = Tensor.of(new int[] {0, 2, -1, 0}, shape);
        Tensor castBool = Tracer.trace(input, t -> t.cast(DataType.BOOL));
        assertEquals(shape, castBool.materialize().shape());
        assertEquals((byte) 0, (byte) TensorTestReads.readValue(castBool, 0, DataType.BOOL));
        assertEquals((byte) 1, (byte) TensorTestReads.readValue(castBool, 1, DataType.BOOL));
        assertEquals((byte) 1, (byte) TensorTestReads.readValue(castBool, 2, DataType.BOOL));
        assertEquals((byte) 0, (byte) TensorTestReads.readValue(castBool, 3, DataType.BOOL));
    }

    @Test
    void canary() {
        Shape shape = Shape.of(2, 2);
        Tensor input0 = Tensor.of(new int[] {0, 2, -1, 0}, shape);
        Tensor input1 = Tensor.of(new int[] {0, 2, -1, 0}, shape);
        // Materialize the reduction result first, then cast
        Tensor sum = Tracer.trace(input0, input1, (t0, t1) -> t0.add(t1).sum(DataType.I32));
        Tensor output = Tracer.trace(sum, s -> s.cast(DataType.FP32));
        assertEquals(Shape.scalar(), output.materialize().shape());
        assertEquals(2.0f, (float) TensorTestReads.readValue(output, 0, DataType.FP32), 0.0001f);
    }

    @Test
    void castsFloatToIntLossy() {
        Shape shape = Shape.of(3);
        Tensor input = Tensor.of(new float[] {1.8f, -2.4f, 0.0f}, shape);

        Tensor castI32 = Tracer.trace(input, t -> t.cast(DataType.I32));
        assertEquals(shape, castI32.materialize().shape());
        assertEquals(1, (int) TensorTestReads.readValue(castI32, 0, DataType.I32));
        assertEquals(-2, (int) TensorTestReads.readValue(castI32, 1, DataType.I32));
        assertEquals(0, (int) TensorTestReads.readValue(castI32, 2, DataType.I32));
    }
}
