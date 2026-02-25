package com.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Indexing;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryAccess;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.memory.impl.DomainFactory;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

class ConcatOpsTest {

    private static MemoryDomain<MemorySegment> domain;

    @BeforeAll
    static void setUpDomain() {
        domain = DomainFactory.ofMemorySegment();
    }

    @Test
    void concatAlongLastAxis() {
        Tensor a = Tensor.of(new float[] {1f, 2f, 3f, 4f}, Shape.of(2, 2));
        Tensor b = Tensor.of(new float[] {5f, 6f, 7f, 8f}, Shape.of(2, 2));

        Tensor out = Tensor.concat(-1, a, b);
        MemoryView<?> view = out.materialize();

        assertEquals(Shape.of(2, 4), view.shape());
        float[] expected = {1f, 2f, 5f, 6f, 3f, 4f, 7f, 8f};
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], readFloat(view, i), 1e-4f);
        }
    }

    @Test
    void concatAlongFirstAxis() {
        Tensor a = Tensor.of(new float[] {1f, 2f, 3f, 4f}, Shape.of(2, 2));
        Tensor b = Tensor.of(new float[] {5f, 6f, 7f, 8f}, Shape.of(2, 2));

        Tensor out = Tensor.concat(0, a, b);
        MemoryView<?> view = out.materialize();

        assertEquals(Shape.of(4, 2), view.shape());
        float[] expected = {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f};
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], readFloat(view, i), 1e-4f);
        }
    }

    @Test
    void concatThreeTensors() {
        Tensor a = Tensor.iota(4, DataType.FP32).view(Shape.of(1, 4));
        Tensor b = Tensor.iota(4, DataType.FP32).add(10f).view(Shape.of(1, 4));
        Tensor c = Tensor.iota(4, DataType.FP32).add(20f).view(Shape.of(1, 4));

        Tensor out = Tensor.concat(1, a, b, c);
        MemoryView<?> view = out.materialize();

        assertEquals(Shape.of(1, 12), view.shape());
        float[] expected = {0f, 1f, 2f, 3f, 10f, 11f, 12f, 13f, 20f, 21f, 22f, 23f};
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], readFloat(view, i), 1e-4f);
        }
    }

    @Test
    void concatSupportsNestedModeLayouts() {
        Tensor a = Tensor.iota(8, DataType.FP32).add(1f).view(Shape.of(2, Shape.of(2, 2)));
        Tensor b = Tensor.iota(8, DataType.FP32).add(9f).view(Shape.of(2, Shape.of(2, 2)));

        Tensor out = Tensor.concat(1, a, b);
        MemoryView<?> view = out.materialize();

        assertEquals(Shape.of(2, 8), view.shape());
        float[] expected = {1f, 2f, 3f, 4f, 9f, 10f, 11f, 12f, 5f, 6f, 7f, 8f, 13f, 14f, 15f, 16f};
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], readFloat(view, i), 1e-4f);
        }
    }

    @Test
    void concatWorksForNonContiguousInputs() {
        Tensor base = Tensor.iota(12, DataType.FP32).view(Shape.of(3, 4));
        Tensor a = base.transpose(0, 1);
        Tensor b = base.add(100f).transpose(0, 1);

        Tensor out = Tensor.concat(1, a, b);
        MemoryView<?> view = out.materialize();

        assertEquals(Shape.of(4, 6), view.shape());
        assertEquals(0f, readFloat(view, 0), 1e-4f);
        assertEquals(4f, readFloat(view, 1), 1e-4f);
        assertEquals(100f, readFloat(view, 3), 1e-4f);
    }

    @Test
    void concatValidatesAxis() {
        Tensor input = Tensor.iota(6, DataType.FP32).view(Shape.of(2, 3));

        assertThrows(IllegalArgumentException.class, () -> Tensor.concat(2, input, input));
        assertThrows(IllegalArgumentException.class, () -> Tensor.concat(-3, input, input));
    }

    @Test
    void concatRejectsRankMismatch() {
        Tensor a = Tensor.iota(6, DataType.FP32).view(Shape.of(2, 3));
        Tensor b = Tensor.iota(6, DataType.FP32);

        assertThrows(IllegalArgumentException.class, () -> Tensor.concat(1, a, b));
    }

    @Test
    void concatRejectsNonAxisDimensionMismatch() {
        Tensor a = Tensor.iota(6, DataType.FP32).view(Shape.of(2, 3));
        Tensor b = Tensor.iota(9, DataType.FP32).view(Shape.of(3, 3));

        assertThrows(IllegalArgumentException.class, () -> Tensor.concat(1, a, b));
    }

    @Test
    void concatRejectsDtypeMismatch() {
        Tensor a = Tensor.iota(6, DataType.FP32).view(Shape.of(2, 3));
        Tensor b = Tensor.iota(6, DataType.I32).view(Shape.of(2, 3));

        assertThrows(IllegalArgumentException.class, () -> Tensor.concat(1, a, b));
    }

    @Test
    void concatWorksThroughTracing() {
        Tensor a = Tensor.of(new float[] {1f, 2f, 3f, 4f}, Shape.of(2, 2));
        Tensor b = Tensor.of(new float[] {5f, 6f, 7f, 8f}, Shape.of(2, 2));

        Tensor traced = Tracer.trace(a, b, (x, y) -> Tensor.concat(1, x, y));
        MemoryView<?> view = traced.materialize();

        assertEquals(Shape.of(2, 4), view.shape());
        float[] expected = {1f, 2f, 5f, 6f, 3f, 4f, 7f, 8f};
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], readFloat(view, i), 1e-4f);
        }
    }

    private static float readFloat(MemoryView<?> view, long linearIndex) {
        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment> typedView = (MemoryView<MemorySegment>) view;
        long offset = Indexing.linearToOffset(typedView, linearIndex);
        MemoryAccess<MemorySegment> access = domain.directAccess();
        return access.readFloat(typedView.memory(), offset);
    }
}
