package com.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Device;
import com.qxotic.jota.Shape;
import com.qxotic.jota.testutil.RunOnAllAvailableBackends;
import com.qxotic.jota.testutil.TensorTestReads;
import java.util.List;
import org.junit.jupiter.api.Test;

@RunOnAllAvailableBackends
class BitwiseOpTest {

    private static final List<DataType> INTEGRAL_TYPES =
            List.of(DataType.I8, DataType.I16, DataType.I32, DataType.I64);

    @Test
    void bitwiseNotWorksForIntegralTypes() {
        Shape shape = Shape.of(4);
        for (DataType dataType : INTEGRAL_TYPES) {
            Tensor input = Tensor.iota(shape.size(), DataType.I64).cast(dataType).view(shape);
            Tensor output = Tracer.trace(input, Tensor::bitwiseNot);

            for (int i = 0; i < shape.size(); i++) {
                long value = readIntegral(output, i, dataType);
                long expected = castIntegral(~i, dataType);
                assertEquals(expected, value, "Mismatch for " + dataType + " at " + i);
            }
        }
    }

    @Test
    void bitwiseBinaryOpsWorkForIntegralTypes() {
        Shape shape = Shape.of(4);
        for (DataType dataType : INTEGRAL_TYPES) {
            Tensor a = Tensor.iota(shape.size(), DataType.I64).cast(dataType).view(shape);
            Tensor b = Tensor.full(3L, shape).cast(dataType);

            Tensor andTensor = Tracer.trace(a, b, Tensor::bitwiseAnd);
            Tensor orTensor = Tracer.trace(a, b, Tensor::bitwiseOr);
            Tensor xorTensor = Tracer.trace(a, b, Tensor::bitwiseXor);

            for (int i = 0; i < shape.size(); i++) {
                long leftValue = castIntegral(i, dataType);
                long rightValue = castIntegral(3, dataType);
                assertEquals(
                        castIntegral((int) leftValue & (int) rightValue, dataType),
                        readIntegral(andTensor, i, dataType),
                        "bitwiseAnd mismatch for " + dataType + " at " + i);
                assertEquals(
                        castIntegral((int) leftValue | (int) rightValue, dataType),
                        readIntegral(orTensor, i, dataType),
                        "bitwiseOr mismatch for " + dataType + " at " + i);
                assertEquals(
                        castIntegral((int) leftValue ^ (int) rightValue, dataType),
                        readIntegral(xorTensor, i, dataType),
                        "bitwiseXor mismatch for " + dataType + " at " + i);
            }
        }
    }

    @Test
    void shiftOpsWorkForIntegralTypesWithNormalizedShiftCounts() {
        Shape shape = Shape.of(4);
        for (DataType dataType : INTEGRAL_TYPES) {
            Tensor a = Tensor.of(new long[] {-8L, -1L, 7L, 64L}, shape).cast(dataType);
            Tensor b = Tensor.of(new long[] {1L, 2L, 31L, 65L}, shape).cast(dataType);

            Tensor leftShift = Tracer.trace(a, b, Tensor::leftShift);
            Tensor rightShift = Tracer.trace(a, b, Tensor::rightShift);
            Tensor rightShiftUnsigned = Tracer.trace(a, b, Tensor::rightShiftUnsigned);

            long[] values = {-8L, -1L, 7L, 64L};
            long[] shifts = {1L, 2L, 31L, 65L};
            for (int i = 0; i < values.length; i++) {
                int shift = normalizedShift(dataType, shifts[i]);
                assertEquals(
                        castIntegral(values[i] << shift, dataType),
                        readIntegral(leftShift, i, dataType),
                        "leftShift mismatch for " + dataType + " at " + i);
                assertEquals(
                        arithmeticRight(values[i], shift, dataType),
                        readIntegral(rightShift, i, dataType),
                        "rightShift mismatch for " + dataType + " at " + i);
                assertEquals(
                        logicalRight(values[i], shift, dataType),
                        readIntegral(rightShiftUnsigned, i, dataType),
                        "rightShiftUnsigned mismatch for " + dataType + " at " + i);
            }
        }
    }

    @Test
    void shiftOpsRejectBoolAndFloat() {
        Shape shape = Shape.of(2);
        Tensor boolTensor = Tensor.full(1L, shape).cast(DataType.BOOL);
        Tensor floatTensor = Tensor.iota(shape.size(), DataType.FP32).view(shape);

        assertThrows(
                IllegalArgumentException.class,
                () -> Tracer.trace(boolTensor, boolTensor, Tensor::leftShift));
        assertThrows(
                IllegalArgumentException.class,
                () -> Tracer.trace(floatTensor, floatTensor, Tensor::leftShift));
        assertThrows(
                IllegalArgumentException.class,
                () -> Tracer.trace(boolTensor, boolTensor, Tensor::rightShift));
        assertThrows(
                IllegalArgumentException.class,
                () -> Tracer.trace(floatTensor, floatTensor, Tensor::rightShift));
        assertThrows(
                IllegalArgumentException.class,
                () -> Tracer.trace(boolTensor, boolTensor, Tensor::rightShiftUnsigned));
        assertThrows(
                IllegalArgumentException.class,
                () -> Tracer.trace(floatTensor, floatTensor, Tensor::rightShiftUnsigned));
    }

    @Test
    void bitwiseOpsRejectBoolAndFloat() {
        Shape shape = Shape.of(2);
        Tensor boolTensor = Tensor.full(1L, shape).cast(DataType.BOOL);
        assertThrows(
                IllegalArgumentException.class, () -> Tracer.trace(boolTensor, Tensor::bitwiseNot));

        Tensor floatTensor = Tensor.iota(shape.size(), DataType.FP32).view(shape);
        assertThrows(
                IllegalArgumentException.class,
                () -> Tracer.trace(floatTensor, Tensor::bitwiseNot));
    }

    private long readIntegral(Tensor tensor, long index, DataType dataType) {
        if (dataType == DataType.I8) {
            return TensorTestReads.readByte(tensor, index);
        }
        if (dataType == DataType.I16) {
            Tensor materialized = tensor.to(Device.PANAMA);
            return (short) TensorTestReads.readLong(materialized.cast(DataType.I64), index);
        }
        if (dataType == DataType.I32) {
            Tensor materialized = tensor.to(Device.PANAMA);
            return (int) TensorTestReads.readLong(materialized.cast(DataType.I64), index);
        }
        if (dataType == DataType.I64) {
            return TensorTestReads.readLong(tensor, index);
        }
        throw new IllegalStateException("Unsupported integral type: " + dataType);
    }

    private long castIntegral(long value, DataType dataType) {
        if (dataType == DataType.I8) {
            return (byte) value;
        }
        if (dataType == DataType.I16) {
            return (short) value;
        }
        if (dataType == DataType.I32) {
            return (int) value;
        }
        if (dataType == DataType.I64) {
            return value;
        }
        throw new IllegalStateException("Unsupported integral type: " + dataType);
    }

    private int normalizedShift(DataType dataType, long shift) {
        if (dataType == DataType.I8) {
            return ((int) shift) & 7;
        }
        if (dataType == DataType.I16) {
            return ((int) shift) & 15;
        }
        if (dataType == DataType.I32) {
            return ((int) shift) & 31;
        }
        return ((int) shift) & 63;
    }

    private long arithmeticRight(long value, int shift, DataType dataType) {
        if (dataType == DataType.I8) {
            return (byte) ((byte) value >> shift);
        }
        if (dataType == DataType.I16) {
            return (short) ((short) value >> shift);
        }
        if (dataType == DataType.I32) {
            return (int) value >> shift;
        }
        return value >> shift;
    }

    private long logicalRight(long value, int shift, DataType dataType) {
        if (dataType == DataType.I8) {
            return (byte) ((((int) (byte) value) & 0xFF) >>> shift);
        }
        if (dataType == DataType.I16) {
            return (short) ((((int) (short) value) & 0xFFFF) >>> shift);
        }
        if (dataType == DataType.I32) {
            return ((int) value) >>> shift;
        }
        return value >>> shift;
    }
}
