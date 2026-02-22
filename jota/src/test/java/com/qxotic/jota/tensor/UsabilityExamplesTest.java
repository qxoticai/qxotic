package com.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Environment;
import com.qxotic.jota.Indexing;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;

class UsabilityExamplesTest {

    @SuppressWarnings("unchecked")
    private static final MemoryDomain<MemorySegment> CONTEXT =
            (MemoryDomain<MemorySegment>) Environment.current().nativeRuntime().memoryDomain();

    @Test
    void elementwiseChainMaterializes() {
        Tensor y = Tensor.iota(6, DataType.FP32).view(Shape.of(2, 3)).add(1.0f).sqrt();

        assertTrue(y.isLazy());
        assertFalse(y.isMaterialized());

        MemoryView<?> output = y.materialize();

        assertEquals(DataType.FP32, output.dataType());
        assertEquals(Shape.of(2, 3), output.shape());
        assertEquals(1.0f, readFloat(output, 0), 0.0001f);
        assertEquals((float) Math.sqrt(6.0), readFloat(output, 5), 0.0001f);
    }

    @Test
    void elementwiseAddSameShape() {
        Tensor y =
                Tensor.iota(6)
                        .cast(DataType.FP32)
                        .view(Shape.of(2, 3))
                        .add(Tensor.ones(Shape.of(2, 3)));

        MemoryView<?> output = y.materialize();

        assertEquals(DataType.FP32, output.dataType());
        assertEquals(Shape.of(2, 3), output.shape());
        assertEquals(1.0f, readFloat(output, 0), 0.0001f);
        assertEquals(3.0f, readFloat(output, 2), 0.0001f);
        assertEquals(4.0f, readFloat(output, 3), 0.0001f);
    }

    @Test
    void transposeViewReordersAxes() {
        Tensor y = Tensor.iota(6).view(Shape.of(2, 3)).transpose(0, 1);

        MemoryView<?> output = y.materialize();

        assertEquals(DataType.I64, output.dataType());
        assertEquals(Shape.of(3, 2), output.shape());
        assertEquals(3L, readLong(output, 1));
        assertEquals(2L, readLong(output, 4));
    }

    @Test
    void sumReducesAlongAxis() {
        Tensor y = Tensor.iota(6, DataType.FP32).view(Shape.of(2, 3)).sum(DataType.FP32, 1);

        MemoryView<?> output = y.materialize();

        assertEquals(DataType.FP32, output.dataType());
        assertEquals(Shape.flat(2), output.shape());
        assertEquals(3.0f, readFloat(output, 0), 0.0001f);
        assertEquals(12.0f, readFloat(output, 1), 0.0001f);
    }

    @Test
    void comparisonProducesBoolTensor() {
        Tensor y = Tensor.iota(5).lessThan(Tensor.full(2L, Shape.flat(5)));

        MemoryView<?> output = y.materialize();

        assertEquals(DataType.BOOL, output.dataType());
        assertEquals(Shape.flat(5), output.shape());
        assertEquals(1, readByte(output, 0));
        assertEquals(1, readByte(output, 1));
        assertEquals(0, readByte(output, 2));
        assertEquals(0, readByte(output, 4));
    }

    private static long readLong(MemoryView<?> view, long linearIndex) {
        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment> typedView = (MemoryView<MemorySegment>) view;
        long offset = Indexing.linearToOffset(typedView, linearIndex);
        return CONTEXT.directAccess().readLong(typedView.memory(), offset);
    }

    private static float readFloat(MemoryView<?> view, long linearIndex) {
        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment> typedView = (MemoryView<MemorySegment>) view;
        long offset = Indexing.linearToOffset(typedView, linearIndex);
        return CONTEXT.directAccess().readFloat(typedView.memory(), offset);
    }

    private static byte readByte(MemoryView<?> view, long linearIndex) {
        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment> typedView = (MemoryView<MemorySegment>) view;
        long offset = Indexing.linearToOffset(typedView, linearIndex);
        return CONTEXT.directAccess().readByte(typedView.memory(), offset);
    }
}
