package com.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Device;
import com.qxotic.jota.Shape;
import com.qxotic.jota.testutil.ConfiguredTestDevice;
import com.qxotic.jota.testutil.RunOnAllAvailableBackends;
import com.qxotic.jota.testutil.TensorTestReads;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

@RunOnAllAvailableBackends
class MatmulOpTest {

    @Test
    void matmulWorks() {
        Tensor output = runMatmul();
        assertEquals(Shape.of(2, 2), output.shape());
        assertClose(readFloat(output, 0), 22f);
        assertClose(readFloat(output, 1), 28f);
        assertClose(readFloat(output, 2), 49f);
        assertClose(readFloat(output, 3), 64f);
    }

    @Test
    void batchedMatmulWorks() {
        Tensor output = runBatchedMatmul();
        assertEquals(Shape.of(2, 2, 2), output.shape());
        assertClose(readFloat(output, 0), 22f);
        assertClose(readFloat(output, 1), 28f);
        assertClose(readFloat(output, 2), 49f);
        assertClose(readFloat(output, 3), 64f);
        assertClose(readFloat(output, 4), 220f);
        assertClose(readFloat(output, 5), 244f);
        assertClose(readFloat(output, 6), 301f);
        assertClose(readFloat(output, 7), 334f);
    }

    @Test
    void matmulHandlesNonContiguousInput() {
        Assumptions.assumeTrue(
                ConfiguredTestDevice.resolve() == Device.PANAMA,
                "Non-contiguous matmul currently panama-only in runtime-agnostic lane");

        Tensor output = runMatmulNonContiguousLeft();
        assertEquals(Shape.of(2, 2), output.shape());
        assertClose(readFloat(output, 0), 35f);
        assertClose(readFloat(output, 1), 44f);
        assertClose(readFloat(output, 2), 44f);
        assertClose(readFloat(output, 3), 56f);
    }

    @Test
    void tracedMatmulHandlesNonContiguousInput() {
        Assumptions.assumeTrue(
                ConfiguredTestDevice.resolve() == Device.PANAMA,
                "Non-contiguous matmul currently panama-only in runtime-agnostic lane");

        Tensor output = runTracedMatmulNonContiguousLeft();
        assertEquals(Shape.of(2, 2), output.shape());
        assertClose(readFloat(output, 0), 35f);
        assertClose(readFloat(output, 1), 44f);
        assertClose(readFloat(output, 2), 44f);
        assertClose(readFloat(output, 3), 56f);
    }

    @Test
    void vectorTimesMatrixIsRejected() {
        assertThrows(
                IllegalArgumentException.class,
                () -> {
                    Tensor vector = Tensor.iota(3, DataType.FP32).add(1f);
                    Tensor matrix = Tensor.iota(6, DataType.FP32).add(1f).view(Shape.of(3, 2));
                    vector.matmul(matrix).materialize();
                });
    }

    @Test
    void matrixTimesVectorIsRejected() {
        assertThrows(
                IllegalArgumentException.class,
                () -> {
                    Tensor matrix = Tensor.iota(6, DataType.FP32).add(1f).view(Shape.of(2, 3));
                    Tensor vector = Tensor.iota(3, DataType.FP32).add(1f);
                    matrix.matmul(vector).materialize();
                });
    }

    @Test
    void scalarMatmulIsRejected() {
        assertThrows(
                IllegalArgumentException.class,
                () -> {
                    Tensor scalar = Tensor.scalar(2f);
                    Tensor matrix = Tensor.iota(6, DataType.FP32).add(1f).view(Shape.of(3, 2));
                    scalar.matmul(matrix).materialize();
                });
    }

    @Test
    void incompatibleInnerDimensionsAreRejected() {
        assertThrows(
                IllegalArgumentException.class,
                () -> {
                    Tensor a = Tensor.iota(6, DataType.FP32).add(1f).view(Shape.of(2, 3));
                    Tensor b = Tensor.iota(8, DataType.FP32).add(1f).view(Shape.of(4, 2));
                    a.matmul(b).materialize();
                });
    }

    @Test
    void incompatibleDtypesAreRejected() {
        assertThrows(
                IllegalArgumentException.class,
                () -> {
                    Tensor a = Tensor.iota(6, DataType.I64).view(Shape.of(2, 3));
                    Tensor b = Tensor.iota(6, DataType.FP32).view(Shape.of(3, 2));
                    a.matmul(b).materialize();
                });
    }

    @Test
    void batchedMatmulIncompatibleBatchIsRejected() {
        assertThrows(
                IllegalArgumentException.class,
                () -> {
                    Tensor a = Tensor.iota(12, DataType.FP32).view(Shape.of(2, 2, 3));
                    Tensor b = Tensor.iota(18, DataType.FP32).view(Shape.of(3, 3, 2));
                    a.batchedMatmul(b).materialize();
                });
    }

    @Test
    void batchedMatmulIncompatibleDtypesAreRejected() {
        assertThrows(
                IllegalArgumentException.class,
                () -> {
                    Tensor a = Tensor.iota(12, DataType.I64).view(Shape.of(2, 2, 3));
                    Tensor b = Tensor.iota(12, DataType.FP32).view(Shape.of(2, 3, 2));
                    a.batchedMatmul(b).materialize();
                });
    }

    private Tensor runMatmul() {
        Tensor a = Tensor.iota(6, DataType.FP32).add(1f).view(Shape.of(2, 3));
        Tensor b = Tensor.iota(6, DataType.FP32).add(1f).view(Shape.of(3, 2));
        Tensor out = a.matmul(b);
        out.materialize();
        return out;
    }

    private Tensor runBatchedMatmul() {
        Tensor a = Tensor.iota(12, DataType.FP32).add(1f).view(Shape.of(2, 2, 3));
        Tensor b = Tensor.iota(12, DataType.FP32).add(1f).view(Shape.of(2, 3, 2));
        Tensor out = a.batchedMatmul(b);
        out.materialize();
        return out;
    }

    private Tensor runMatmulNonContiguousLeft() {
        Tensor a = Tensor.iota(6, DataType.FP32).add(1f).view(Shape.of(3, 2)).transpose(-2, -1);
        Tensor b = Tensor.iota(6, DataType.FP32).add(1f).view(Shape.of(3, 2));
        Tensor out = a.matmul(b);
        out.materialize();
        return out;
    }

    private Tensor runTracedMatmulNonContiguousLeft() {
        Tensor a = Tensor.iota(6, DataType.FP32).add(1f).view(Shape.of(3, 2)).transpose(-2, -1);
        Tensor b = Tensor.iota(6, DataType.FP32).add(1f).view(Shape.of(3, 2));
        Tensor out = Tracer.trace(a, b, Tensor::matmul);
        out.materialize();
        return out;
    }

    private static float readFloat(Tensor tensor, long linearIndex) {
        return TensorTestReads.readFloat(tensor, linearIndex);
    }

    private static void assertClose(float actual, float expected) {
        assertEquals(expected, actual, 1e-4f);
    }
}
