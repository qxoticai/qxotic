package com.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.random.RandomKey;
import org.junit.jupiter.api.Test;

class TensorStaticFactoriesCoverageTest {

    @Test
    void callsEveryPublicStaticMethodOnTensor() {
        Tensor a = Tensor.of(new float[] {1f, 2f, 3f, 4f}, Shape.of(2, 2));
        Tensor b = Tensor.of(new float[] {5f, 6f, 7f, 8f}, Shape.of(2, 2));
        Tensor c = Tensor.of(new float[] {9f, 10f, 11f, 12f}, Shape.of(2, 2));

        // Static structural methods.
        assertTensor(Tensor.concat(0, a, b), DataType.FP32, Shape.of(4, 2));
        assertTensor(Tensor.stack(0, a, b, c), DataType.FP32, Shape.of(3, 2, 2));
        Tensor[] split = Tensor.split(1, a, 1, 1);
        assertEquals(2, split.length);
        assertTensor(split[0], DataType.FP32, Shape.of(2, 1));
        assertTensor(split[1], DataType.FP32, Shape.of(2, 1));

        // Core creation.
        MemoryView<?> view = a.materialize();
        assertTensor(Tensor.of(view), DataType.FP32, Shape.of(2, 2));

        // Zero/one/iota.
        assertTensor(Tensor.zeros(Shape.of(2, 3)), DataType.defaultFloat(), Shape.of(2, 3));
        assertTensor(Tensor.zeros(DataType.I32, Shape.of(2, 3)), DataType.I32, Shape.of(2, 3));
        assertTensor(Tensor.ones(Shape.of(2, 3)), DataType.defaultFloat(), Shape.of(2, 3));
        assertTensor(Tensor.ones(DataType.I64, Shape.of(2, 3)), DataType.I64, Shape.of(2, 3));
        assertTensor(Tensor.iota(5), DataType.I64, Shape.of(5));
        assertTensor(Tensor.iota(5, DataType.FP32), DataType.FP32, Shape.of(5));

        RandomKey key = RandomKey.of(99L);

        // rand overloads.
        assertTensor(Tensor.rand(key, 4, DataType.FP64), DataType.FP64, Shape.of(4));
        assertTensor(
                Tensor.rand(key, Shape.of(2, 2), DataType.FP64), DataType.FP64, Shape.of(2, 2));

        // randn overloads.
        assertTensor(Tensor.randn(key, 4, DataType.FP64), DataType.FP64, Shape.of(4));
        assertTensor(
                Tensor.randn(key, Shape.of(2, 2), DataType.FP64), DataType.FP64, Shape.of(2, 2));

        // randInt overloads.
        assertTensor(Tensor.randInt(key, 0, 10, 4, DataType.I64), DataType.I64, Shape.of(4));
        assertTensor(
                Tensor.randInt(key, 0, 10, Shape.of(2, 2), DataType.I64),
                DataType.I64,
                Shape.of(2, 2));

        // uniform overloads.
        assertTensor(Tensor.uniform(key, 4, -1.0, 1.0, DataType.FP64), DataType.FP64, Shape.of(4));
        assertTensor(
                Tensor.uniform(key, Shape.of(2, 2), -1.0, 1.0, DataType.FP64),
                DataType.FP64,
                Shape.of(2, 2));

        // normal overloads.
        assertTensor(Tensor.normal(key, 4, 0.0, 1.0, DataType.FP64), DataType.FP64, Shape.of(4));
        assertTensor(
                Tensor.normal(key, Shape.of(2, 2), 0.0, 1.0, DataType.FP64),
                DataType.FP64,
                Shape.of(2, 2));

        // uniformInt overloads.
        assertTensor(Tensor.uniformInt(key, 0, 10, 4, DataType.I64), DataType.I64, Shape.of(4));
        assertTensor(
                Tensor.uniformInt(key, 0, 10, Shape.of(2, 2), DataType.I64),
                DataType.I64,
                Shape.of(2, 2));

        // normalInt overloads.
        assertTensor(Tensor.normalInt(key, 4, 0.0, 1.0, DataType.I64), DataType.I64, Shape.of(4));
        assertTensor(
                Tensor.normalInt(key, Shape.of(2, 2), 0.0, 1.0, DataType.I64),
                DataType.I64,
                Shape.of(2, 2));

        // full overloads.
        assertTensor(Tensor.full(1.5f, Shape.of(2, 2)), DataType.FP32, Shape.of(2, 2));
        assertTensor(Tensor.full(1.5, Shape.of(2, 2)), DataType.FP64, Shape.of(2, 2));
        assertTensor(Tensor.full(5L, Shape.of(2, 2)), DataType.I64, Shape.of(2, 2));
        assertTensor(Tensor.full(5, Shape.of(2, 2)), DataType.I32, Shape.of(2, 2));
        assertTensor(Tensor.full(7, DataType.I16, Shape.of(2, 2)), DataType.I16, Shape.of(2, 2));

        // scalar overloads.
        assertTensor(Tensor.scalar(1), DataType.I32, Shape.scalar());
        assertTensor(Tensor.scalar(1.0f), DataType.FP32, Shape.scalar());
        assertTensor(Tensor.scalar(1.0), DataType.FP64, Shape.scalar());
        assertTensor(Tensor.scalar(1L), DataType.I64, Shape.scalar());
        assertTensor(Tensor.scalar(1.0, DataType.FP32), DataType.FP32, Shape.scalar());
        assertTensor(Tensor.scalar(1L, DataType.I16), DataType.I16, Shape.scalar());

        // Array-backed overloads.
        assertTensor(Tensor.of(new float[] {1f, 2f}), DataType.FP32, Shape.of(2));
        assertTensor(Tensor.of(new float[] {1f, 2f}, Shape.of(2)), DataType.FP32, Shape.of(2));
        assertTensor(Tensor.of(new double[] {1, 2}), DataType.FP64, Shape.of(2));
        assertTensor(Tensor.of(new double[] {1, 2}, Shape.of(2)), DataType.FP64, Shape.of(2));
        assertTensor(Tensor.of(new int[] {1, 2}), DataType.I32, Shape.of(2));
        assertTensor(Tensor.of(new int[] {1, 2}, Shape.of(2)), DataType.I32, Shape.of(2));
        assertTensor(Tensor.of(new long[] {1, 2}), DataType.I64, Shape.of(2));
        assertTensor(Tensor.of(new long[] {1, 2}, Shape.of(2)), DataType.I64, Shape.of(2));
        assertTensor(Tensor.of(new boolean[] {true, false}), DataType.BOOL, Shape.of(2));
        assertTensor(
                Tensor.of(new boolean[] {true, false}, Shape.of(2)), DataType.BOOL, Shape.of(2));
    }

    private static void assertTensor(Tensor tensor, DataType dataType, Shape shape) {
        assertEquals(dataType, tensor.dataType());
        assertEquals(shape, tensor.shape());
        MemoryView<?> view = tensor.materialize();
        assertEquals(dataType, view.dataType());
        assertEquals(shape, view.shape());
    }
}
