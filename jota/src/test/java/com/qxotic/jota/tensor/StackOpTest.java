package com.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.testutil.RunOnAllAvailableBackends;
import com.qxotic.jota.testutil.TensorTestReads;
import org.junit.jupiter.api.Test;

@RunOnAllAvailableBackends
class StackOpTest {

    @Test
    void stackAlongFirstAxis() {
        Tensor a = Tensor.of(new float[] {1f, 2f, 3f, 4f}, Shape.of(2, 2));
        Tensor b = Tensor.of(new float[] {5f, 6f, 7f, 8f}, Shape.of(2, 2));

        Tensor out = Tensor.stack(0, a, b);
        MemoryView<?> view = out.materialize();

        assertEquals(Shape.of(2, 2, 2), view.shape());
        float[] expected = {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f};
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], TensorTestReads.readFloat(out, i), 1e-4f);
        }
    }

    @Test
    void stackAlongLastAxis() {
        Tensor a = Tensor.of(new float[] {1f, 2f, 3f, 4f}, Shape.of(2, 2));
        Tensor b = Tensor.of(new float[] {5f, 6f, 7f, 8f}, Shape.of(2, 2));

        Tensor out = Tensor.stack(-1, a, b);
        MemoryView<?> view = out.materialize();

        assertEquals(Shape.of(2, 2, 2), view.shape());
        float[] expected = {1f, 5f, 2f, 6f, 3f, 7f, 4f, 8f};
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], TensorTestReads.readFloat(out, i), 1e-4f);
        }
    }

    @Test
    void stackThreeTensors() {
        Tensor a = Tensor.iota(4, DataType.FP32).view(Shape.of(2, 2));
        Tensor b = a.add(10f);
        Tensor c = a.add(20f);

        Tensor out = Tensor.stack(1, a, b, c);
        MemoryView<?> view = out.materialize();

        assertEquals(Shape.of(2, 3, 2), view.shape());
        assertEquals(0f, TensorTestReads.readFloat(out, 0), 1e-4f);
        assertEquals(1f, TensorTestReads.readFloat(out, 1), 1e-4f);
        assertEquals(10f, TensorTestReads.readFloat(out, 2), 1e-4f);
    }

    @Test
    void stackSupportsNestedModeLayouts() {
        Tensor a = Tensor.iota(8, DataType.FP32).add(1f).view(Shape.of(2, Shape.of(2, 2)));
        Tensor b = Tensor.iota(8, DataType.FP32).add(9f).view(Shape.of(2, Shape.of(2, 2)));

        Tensor out = Tensor.stack(0, a, b);
        MemoryView<?> view = out.materialize();

        assertEquals(Shape.of(2, 2, 4), view.shape());
    }

    @Test
    void stackValidatesAxis() {
        Tensor a = Tensor.iota(6, DataType.FP32).view(Shape.of(2, 3));
        Tensor b = Tensor.iota(6, DataType.FP32).view(Shape.of(2, 3));

        assertThrows(IllegalArgumentException.class, () -> Tensor.stack(3, a, b));
        assertThrows(IllegalArgumentException.class, () -> Tensor.stack(-4, a, b));
    }

    @Test
    void stackRejectsShapeMismatch() {
        Tensor a = Tensor.iota(6, DataType.FP32).view(Shape.of(2, 3));
        Tensor b = Tensor.iota(9, DataType.FP32).view(Shape.of(3, 3));

        assertThrows(IllegalArgumentException.class, () -> Tensor.stack(0, a, b));
    }

    @Test
    void stackRejectsDtypeMismatch() {
        Tensor a = Tensor.iota(6, DataType.FP32).view(Shape.of(2, 3));
        Tensor b = Tensor.iota(6, DataType.I32).view(Shape.of(2, 3));

        assertThrows(IllegalArgumentException.class, () -> Tensor.stack(0, a, b));
    }

    @Test
    void stackWorksThroughTracing() {
        Tensor a = Tensor.of(new float[] {1f, 2f, 3f, 4f}, Shape.of(2, 2));
        Tensor b = Tensor.of(new float[] {5f, 6f, 7f, 8f}, Shape.of(2, 2));

        Tensor traced = Tracer.trace(a, b, (x, y) -> Tensor.stack(0, x, y));
        MemoryView<?> view = traced.materialize();

        assertEquals(Shape.of(2, 2, 2), view.shape());
    }
}
