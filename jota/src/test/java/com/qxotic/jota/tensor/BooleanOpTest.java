package com.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import com.qxotic.jota.testutil.RunOnAllAvailableBackends;
import com.qxotic.jota.testutil.TensorTestReads;
import java.util.List;
import org.junit.jupiter.api.Test;

@RunOnAllAvailableBackends
class BooleanOpTest {

    private static final DataType[] NON_BOOL_TYPES =
            new DataType[] {
                DataType.I8,
                DataType.I16,
                DataType.I32,
                DataType.I64,
                DataType.FP16,
                DataType.BF16,
                DataType.FP32,
                DataType.FP64
            };

    private static final DataType[] PRIMITIVE_DATA_TYPES =
            new DataType[] {
                DataType.BOOL,
                DataType.I8,
                DataType.I16,
                DataType.I32,
                DataType.I64,
                DataType.FP16,
                DataType.BF16,
                DataType.FP32,
                DataType.FP64
            };

    @Test
    void logicalOpsWorkOnBool() {
        Shape shape = Shape.of(4);
        Tensor a = boolPattern(shape, new long[] {1, 0, 1, 0});
        Tensor b = boolPattern(shape, new long[] {1, 1, 0, 0});

        Tensor notA = Tracer.trace(a, Tensor::logicalNot);
        Tensor andTensor = Tracer.trace(a, b, Tensor::logicalAnd);
        Tensor orTensor = Tracer.trace(a, b, Tensor::logicalOr);
        Tensor xorTensor = Tracer.trace(a, b, Tensor::logicalXor);

        notA.materialize();
        andTensor.materialize();
        orTensor.materialize();
        xorTensor.materialize();

        assertEquals(0, readBool(notA, 0));
        assertEquals(1, readBool(notA, 1));
        assertEquals(0, readBool(notA, 2));
        assertEquals(1, readBool(notA, 3));

        assertEquals(1, readBool(andTensor, 0));
        assertEquals(0, readBool(andTensor, 1));
        assertEquals(0, readBool(andTensor, 2));
        assertEquals(0, readBool(andTensor, 3));

        assertEquals(1, readBool(orTensor, 0));
        assertEquals(1, readBool(orTensor, 1));
        assertEquals(1, readBool(orTensor, 2));
        assertEquals(0, readBool(orTensor, 3));

        assertEquals(0, readBool(xorTensor, 0));
        assertEquals(1, readBool(xorTensor, 1));
        assertEquals(1, readBool(xorTensor, 2));
        assertEquals(0, readBool(xorTensor, 3));
    }

    @Test
    void comparisonsSupportAllTypes() {
        Shape shape = Shape.of(4);
        for (DataType dataType : PRIMITIVE_DATA_TYPES) {
            Tensor leftTensor = range(dataType, shape);
            Tensor rightTensor = range(dataType, shape);

            Tensor equalTensor = Tracer.trace(leftTensor, rightTensor, Tensor::equal);
            equalTensor.materialize();
            for (int i = 0; i < shape.size(); i++) {
                assertEquals(1, readBool(equalTensor, i));
            }

            Tensor thresholdTensor =
                    dataType == DataType.BOOL
                            ? boolPattern(shape, new long[] {1, 1, 0, 0})
                            : Tensor.full(2L, dataType, shape);
            Tensor lessThanTensor = Tracer.trace(leftTensor, thresholdTensor, Tensor::lessThan);
            lessThanTensor.materialize();

            if (dataType == DataType.BOOL) {
                assertEquals(1, readBool(lessThanTensor, 0));
                assertEquals(0, readBool(lessThanTensor, 1));
                assertEquals(0, readBool(lessThanTensor, 2));
                assertEquals(0, readBool(lessThanTensor, 3));
            } else {
                assertEquals(1, readBool(lessThanTensor, 0));
                assertEquals(1, readBool(lessThanTensor, 1));
                assertEquals(0, readBool(lessThanTensor, 2));
                assertEquals(0, readBool(lessThanTensor, 3));
            }
        }
    }

    @Test
    void fusedWhere() {
        Shape shape = Shape.of(4);
        Tensor trueTensor = Tensor.iota(shape.size(), DataType.I32).view(shape);
        Tensor falseTensor = Tensor.full(2L, DataType.I32, shape);

        Tensor min =
                Tracer.trace(
                        trueTensor,
                        falseTensor,
                        // Odd min function.
                        (t, f) -> t.lessThan(f).where(t, f));

        min.materialize();
        assertEquals(0, readInt(min, 0));
        assertEquals(1, readInt(min, 1));
        assertEquals(2, readInt(min, 2));
        assertEquals(2, readInt(min, 3));
    }

    @Test
    void whereSelectsBetweenValues() {
        Shape shape = Shape.of(4);
        Tensor condition = boolPattern(shape, new long[] {1, 0, 1, 0});
        Tensor trueTensor = Tensor.iota(shape.size(), DataType.I32).view(shape);
        Tensor falseTensor = Tensor.full(-1L, DataType.I32, shape);

        Tensor selected =
                Tracer.trace(
                        List.of(condition, trueTensor, falseTensor),
                        tensors -> tensors.get(0).where(tensors.get(1), tensors.get(2)));
        selected.materialize();

        assertEquals(0, readInt(selected, 0));
        assertEquals(-1, readInt(selected, 1));
        assertEquals(2, readInt(selected, 2));
        assertEquals(-1, readInt(selected, 3));
    }

    @Test
    void logicalXorRejectsNonBoolTypes() {
        Shape shape = Shape.of(4);
        for (DataType dataType : NON_BOOL_TYPES) {
            Tensor a = range(dataType, shape);
            Tensor b = range(dataType, shape);
            assertThrows(
                    IllegalArgumentException.class,
                    () -> Tracer.trace(a, b, Tensor::logicalXor),
                    "Expected logicalXor to reject " + dataType);
        }
    }

    @Test
    void anyAndAllHandleBoolPatterns() {
        Shape shape = Shape.of(4);
        Tensor allFalse = boolPattern(shape, new long[] {0, 0, 0, 0});
        Tensor mixed = boolPattern(shape, new long[] {0, 1, 0, 0});
        Tensor allTrue = boolPattern(shape, new long[] {1, 1, 1, 1});

        assertEquals(0, readBool(allFalse.any(), 0));
        assertEquals(0, readBool(allFalse.all(), 0));

        assertEquals(1, readBool(mixed.any(), 0));
        assertEquals(0, readBool(mixed.all(), 0));

        assertEquals(1, readBool(allTrue.any(), 0));
        assertEquals(1, readBool(allTrue.all(), 0));
    }

    @Test
    void anyAndAllRejectNonBoolTypes() {
        Shape shape = Shape.of(4);
        for (DataType dataType : NON_BOOL_TYPES) {
            Tensor input = range(dataType, shape);
            assertThrows(
                    IllegalArgumentException.class,
                    input::any,
                    "Expected any() to reject " + dataType);
            assertThrows(
                    IllegalArgumentException.class,
                    input::all,
                    "Expected all() to reject " + dataType);
        }
    }

    private Tensor range(DataType dataType, Shape shape) {
        if (dataType == DataType.BOOL) {
            return boolPattern(shape, new long[] {0, 1, 0, 1});
        }
        return Tensor.iota(shape.size(), DataType.I64).cast(dataType).view(shape);
    }

    private Tensor boolPattern(Shape shape, long[] values) {
        return Tensor.of(values).cast(DataType.BOOL).view(shape);
    }

    private byte readBool(Tensor tensor, long linearIndex) {
        return TensorTestReads.readByte(tensor, linearIndex);
    }

    private int readInt(Tensor tensor, long linearIndex) {
        return (int) TensorTestReads.readValue(tensor, linearIndex, DataType.I32);
    }
}
