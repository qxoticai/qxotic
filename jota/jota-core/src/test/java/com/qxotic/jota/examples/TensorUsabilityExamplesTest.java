package com.qxotic.jota.examples;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.tensor.Tensor;
import com.qxotic.jota.tensor.TensorTestInternals;
import com.qxotic.jota.testutil.RunOnAllAvailableBackends;
import com.qxotic.jota.testutil.TensorTestReads;
import org.junit.jupiter.api.Test;

@RunOnAllAvailableBackends
class TensorUsabilityExamplesTest {

    @Test
    void elementwiseChainMaterializes() {
        Tensor y = Tensor.iota(6, DataType.FP32).view(Shape.of(2, 3)).add(1.0f).sqrt();

        assertTrue(TensorTestInternals.isLazy(y));
        assertFalse(TensorTestInternals.isMaterialized(y));

        MemoryView<?> output = y.materialize();

        assertEquals(DataType.FP32, output.dataType());
        assertEquals(Shape.of(2, 3), output.shape());
        assertEquals(1.0f, TensorTestReads.readFloat(y, 0), 0.0001f);
        assertEquals((float) Math.sqrt(6.0), TensorTestReads.readFloat(y, 5), 0.0001f);
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
        assertEquals(1.0f, TensorTestReads.readFloat(y, 0), 0.0001f);
        assertEquals(3.0f, TensorTestReads.readFloat(y, 2), 0.0001f);
        assertEquals(4.0f, TensorTestReads.readFloat(y, 3), 0.0001f);
    }

    @Test
    void transposeViewReordersAxes() {
        Tensor y = Tensor.iota(6).view(Shape.of(2, 3)).transpose(0, 1);

        MemoryView<?> output = y.materialize();

        assertEquals(DataType.I64, output.dataType());
        assertEquals(Shape.of(3, 2), output.shape());
        assertEquals(3L, TensorTestReads.readLong(y, 1));
        assertEquals(2L, TensorTestReads.readLong(y, 4));
    }

    @Test
    void sumReducesAlongAxis() {
        Tensor y = Tensor.iota(6, DataType.FP32).view(Shape.of(2, 3)).sum(DataType.FP32, 1);

        MemoryView<?> output = y.materialize();

        assertEquals(DataType.FP32, output.dataType());
        assertEquals(Shape.flat(2), output.shape());
        assertEquals(3.0f, TensorTestReads.readFloat(y, 0), 0.0001f);
        assertEquals(12.0f, TensorTestReads.readFloat(y, 1), 0.0001f);
    }

    @Test
    void comparisonProducesBoolTensor() {
        Tensor y = Tensor.iota(5).lessThan(Tensor.full(2L, Shape.flat(5)));

        MemoryView<?> output = y.materialize();

        assertEquals(DataType.BOOL, output.dataType());
        assertEquals(Shape.flat(5), output.shape());
        assertEquals(1, TensorTestReads.readByte(y, 0));
        assertEquals(1, TensorTestReads.readByte(y, 1));
        assertEquals(0, TensorTestReads.readByte(y, 2));
        assertEquals(0, TensorTestReads.readByte(y, 4));
    }
}
