package com.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jota.DataType;
import com.qxotic.jota.DeviceType;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.testutil.ConfiguredTestDevice;
import com.qxotic.jota.testutil.RunOnAllAvailableBackends;
import com.qxotic.jota.testutil.TensorTestReads;
import java.util.List;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

@RunOnAllAvailableBackends
class ShiftOpTest {

    private static final List<DataType> INTEGRAL_TYPES =
            List.of(DataType.I8, DataType.I16, DataType.I32, DataType.I64);

    private static final List<DataType> NON_INTEGRAL_OR_BOOL_TYPES =
            List.of(DataType.BOOL, DataType.FP16, DataType.BF16, DataType.FP32, DataType.FP64);

    @Test
    void commonI32ShiftCasesMatchJavaSemantics() {
        Tensor values = Tensor.of(new int[] {-8, -1, 7, 64});
        Tensor shifts = Tensor.of(new int[] {1, 2, 3, 4});

        Tensor left = values.leftShift(shifts);
        Tensor right = values.rightShift(shifts);
        Tensor rightUnsigned = values.rightShiftUnsigned(shifts);

        int[] valueArray = {-8, -1, 7, 64};
        int[] shiftArray = {1, 2, 3, 4};
        for (int i = 0; i < valueArray.length; i++) {
            int s = shiftArray[i] & 31;
            assertEquals(valueArray[i] << s, readInt(left, i));
            assertEquals(valueArray[i] >> s, readInt(right, i));
            assertEquals(valueArray[i] >>> s, readInt(rightUnsigned, i));
        }
    }

    @Test
    void unsignedRightShiftDiffersFromArithmeticForNegativeValues() {
        Tensor values = Tensor.of(new int[] {-1, -8, -1024});
        Tensor one = Tensor.of(new int[] {1, 1, 1});

        Tensor arithmetic = values.rightShift(one);
        Tensor logical = values.rightShiftUnsigned(one);

        assertEquals(-1, readInt(arithmetic, 0));
        assertEquals(Integer.MAX_VALUE, readInt(logical, 0));

        assertEquals(-4, readInt(arithmetic, 1));
        assertEquals(2147483644, readInt(logical, 1));
    }

    @Test
    void shiftCountsAreNormalizedForAllIntegralTypes() {
        long[] values = {-8L, -1L, 7L, 64L};
        long[] counts = {0L, 8L, -1L, 65L};
        Shape shape = Shape.of(values.length);

        for (DataType type : INTEGRAL_TYPES) {
            Tensor valueTensor = Tensor.of(values, shape).cast(type);
            Tensor countTensor = Tensor.of(counts, shape).cast(type);

            Tensor left = valueTensor.leftShift(countTensor);
            Tensor right = valueTensor.rightShift(countTensor);
            Tensor rightUnsigned = valueTensor.rightShiftUnsigned(countTensor);

            for (int i = 0; i < values.length; i++) {
                int s = normalizedShift(type, counts[i]);
                assertEquals(
                        expectedLeft(values[i], s, type),
                        readIntegral(left, i, type),
                        "leftShift mismatch for " + type + " at " + i);
                assertEquals(
                        expectedRight(values[i], s, type),
                        readIntegral(right, i, type),
                        "rightShift mismatch for " + type + " at " + i);
                assertEquals(
                        expectedRightUnsigned(values[i], s, type),
                        readIntegral(rightUnsigned, i, type),
                        "rightShiftUnsigned mismatch for " + type + " at " + i);
            }
        }
    }

    @Test
    void shiftSupportsScalarBroadcastOnEitherOperand() {
        Tensor values = Tensor.of(new int[] {1, 2, 3, 4});
        Tensor two = Tensor.scalar(2, DataType.I32);

        Tensor left = values.leftShift(two);
        assertEquals(4, readInt(left, 0));
        assertEquals(8, readInt(left, 1));

        Tensor base = Tensor.scalar(64, DataType.I32);
        Tensor counts = Tensor.of(new int[] {0, 1, 2, 3});
        Tensor right = base.rightShiftUnsigned(counts);
        assertEquals(64, readInt(right, 0));
        assertEquals(32, readInt(right, 1));
        assertEquals(16, readInt(right, 2));
        assertEquals(8, readInt(right, 3));
    }

    @Test
    void shiftWorksThroughTracing() {
        Tensor values = Tensor.of(new int[] {4, 8, 16, 32});
        Tensor counts = Tensor.of(new int[] {1, 1, 2, 3});

        Tensor tracedLeft = Tracer.trace(values, counts, Tensor::leftShift);
        Tensor tracedRight = Tracer.trace(values, counts, Tensor::rightShiftUnsigned);

        assertEquals(8, readInt(tracedLeft, 0));
        assertEquals(16, readInt(tracedLeft, 1));
        assertEquals(64, readInt(tracedLeft, 2));
        assertEquals(256, readInt(tracedLeft, 3));

        assertEquals(2, readInt(tracedRight, 0));
        assertEquals(4, readInt(tracedRight, 1));
        assertEquals(4, readInt(tracedRight, 2));
        assertEquals(4, readInt(tracedRight, 3));
    }

    @Test
    void shiftWorksForNonContiguousInputs() {
        Assumptions.assumeTrue(
                ConfiguredTestDevice.resolve().belongsTo(DeviceType.PANAMA),
                "Non-contiguous shift currently panama-only in runtime-agnostic lane");

        Tensor base = Tensor.iota(12, DataType.I32).view(Shape.of(3, 4));
        Tensor values = base.transpose(0, 1);
        Tensor ones = Tensor.full(1, DataType.I32, values.shape());

        Tensor out = values.leftShift(ones);
        MemoryView<?> outView = out.materialize();
        assertEquals(Shape.of(4, 3), outView.shape());
        assertEquals(0, readInt(out, 0));
        assertEquals(8, readInt(out, 1));
        assertEquals(16, readInt(out, 2));
    }

    @Test
    void shiftOpsPreserveIntegralDType() {
        for (DataType type : INTEGRAL_TYPES) {
            Tensor values = Tensor.of(new long[] {1L, 2L, 3L, 4L}).cast(type);
            Tensor counts = Tensor.of(new long[] {1L, 1L, 1L, 1L}).cast(type);

            assertEquals(type, values.leftShift(counts).materialize().dataType());
            assertEquals(type, values.rightShift(counts).materialize().dataType());
            assertEquals(type, values.rightShiftUnsigned(counts).materialize().dataType());
        }
    }

    @Test
    void i32ShiftCountsAreAcceptedForAllIntegralValueTypes() {
        Tensor i32Counts = Tensor.of(new int[] {1, 2, 3, 4});
        for (DataType valueType : INTEGRAL_TYPES) {
            Tensor values = Tensor.of(new long[] {16L, 32L, 64L, 128L}).cast(valueType);
            MemoryView<?> shifted = values.rightShift(i32Counts).materialize();
            assertEquals(valueType, shifted.dataType());
        }
    }

    @Test
    void nonI32IntegralShiftCountsAreAcceptedAndCastInternally() {
        Tensor values = Tensor.of(new long[] {32L, 64L, 128L, 256L}).cast(DataType.I64);
        Tensor countsI8 = Tensor.of(new long[] {1L, 2L, 3L, 4L}).cast(DataType.I8);
        Tensor countsI16 = Tensor.of(new long[] {1L, 2L, 3L, 4L}).cast(DataType.I16);
        Tensor countsI64 = Tensor.of(new long[] {1L, 2L, 3L, 4L}).cast(DataType.I64);

        Tensor outI8 = values.rightShift(countsI8);
        Tensor outI16 = values.rightShift(countsI16);
        Tensor outI64 = values.rightShift(countsI64);

        assertEquals(16L, readIntegral(outI8, 0, DataType.I64));
        assertEquals(16L, readIntegral(outI16, 1, DataType.I64));
        assertEquals(16L, readIntegral(outI64, 2, DataType.I64));
    }

    @Test
    void exhaustiveIntegralValueAndShiftTypeMatrixProducesExpectedResults() {
        long[] values = {-9L, -1L, 0L, 1L, 7L, 33L};
        long[] shiftCounts = {0L, 1L, 2L, 31L, 63L, -1L};
        Shape shape = Shape.of(values.length);

        for (DataType valueType : INTEGRAL_TYPES) {
            Tensor valueTensor = Tensor.of(values, shape).cast(valueType);
            for (DataType shiftType : INTEGRAL_TYPES) {
                Tensor shiftTensor = Tensor.of(shiftCounts, shape).cast(shiftType);

                Tensor left = valueTensor.leftShift(shiftTensor);
                Tensor right = valueTensor.rightShift(shiftTensor);
                Tensor rightUnsigned = valueTensor.rightShiftUnsigned(shiftTensor);

                for (int i = 0; i < values.length; i++) {
                    int castedCount = (int) shiftCounts[i];
                    int normalized = normalizedShift(valueType, castedCount);
                    assertEquals(
                            expectedLeft(values[i], normalized, valueType),
                            readIntegral(left, i, valueType),
                            "leftShift mismatch for value="
                                    + valueType
                                    + ", shift="
                                    + shiftType
                                    + ", i="
                                    + i);
                    assertEquals(
                            expectedRight(values[i], normalized, valueType),
                            readIntegral(right, i, valueType),
                            "rightShift mismatch for value="
                                    + valueType
                                    + ", shift="
                                    + shiftType
                                    + ", i="
                                    + i);
                    assertEquals(
                            expectedRightUnsigned(values[i], normalized, valueType),
                            readIntegral(rightUnsigned, i, valueType),
                            "rightShiftUnsigned mismatch for value="
                                    + valueType
                                    + ", shift="
                                    + shiftType
                                    + ", i="
                                    + i);
                }
            }
        }
    }

    @Test
    void nonIntegralOrBoolShiftCountsAreRejectedForIntegralValues() {
        Tensor values = Tensor.of(new int[] {8, 16, 32, 64});
        for (DataType shiftType : NON_INTEGRAL_OR_BOOL_TYPES) {
            Tensor shiftCounts = Tensor.of(new long[] {1L, 1L, 1L, 1L}).cast(shiftType);
            assertThrows(
                    IllegalArgumentException.class,
                    () -> values.leftShift(shiftCounts),
                    () -> "leftShift should reject shift dtype " + shiftType);
            assertThrows(
                    IllegalArgumentException.class,
                    () -> values.rightShift(shiftCounts),
                    () -> "rightShift should reject shift dtype " + shiftType);
            assertThrows(
                    IllegalArgumentException.class,
                    () -> values.rightShiftUnsigned(shiftCounts),
                    () -> "rightShiftUnsigned should reject shift dtype " + shiftType);
        }
    }

    @Test
    void nonIntegralOrBoolValueTypesAreRejectedEvenWithIntegralShiftCounts() {
        Tensor shiftCounts = Tensor.of(new int[] {1, 1, 1, 1});
        for (DataType valueType : NON_INTEGRAL_OR_BOOL_TYPES) {
            Tensor values = Tensor.of(new long[] {1L, 2L, 3L, 4L}).cast(valueType);
            assertThrows(
                    IllegalArgumentException.class,
                    () -> values.leftShift(shiftCounts),
                    () -> "leftShift should reject value dtype " + valueType);
            assertThrows(
                    IllegalArgumentException.class,
                    () -> values.rightShift(shiftCounts),
                    () -> "rightShift should reject value dtype " + valueType);
            assertThrows(
                    IllegalArgumentException.class,
                    () -> values.rightShiftUnsigned(shiftCounts),
                    () -> "rightShiftUnsigned should reject value dtype " + valueType);
        }
    }

    @Test
    void extremeShiftCountsFollowInt32CastAndNormalization() {
        long[] counts = {Integer.MIN_VALUE, Integer.MAX_VALUE, Long.MIN_VALUE, Long.MAX_VALUE};
        long[] valuesRaw = {1L, -1L, 64L, -64L};
        Tensor shiftsI64 = Tensor.of(counts, Shape.of(4)).cast(DataType.I64);

        for (DataType valueType : INTEGRAL_TYPES) {
            Tensor values = Tensor.of(valuesRaw, Shape.of(4)).cast(valueType);
            Tensor out = values.leftShift(shiftsI64);
            for (int i = 0; i < counts.length; i++) {
                int normalized = normalizedShift(valueType, (int) counts[i]);
                long expected = expectedLeft(valuesRaw[i], normalized, valueType);
                assertEquals(expected, readIntegral(out, i, valueType));
            }
        }
    }

    @Test
    void shiftOpsRejectBoolAndFloatingTypes() {
        Tensor boolTensor = Tensor.scalar(1, DataType.BOOL);
        Tensor fpTensor = Tensor.scalar(1.0, DataType.FP32);

        assertThrows(IllegalArgumentException.class, () -> boolTensor.leftShift(boolTensor));
        assertThrows(IllegalArgumentException.class, () -> fpTensor.leftShift(fpTensor));

        assertThrows(IllegalArgumentException.class, () -> boolTensor.rightShift(boolTensor));
        assertThrows(IllegalArgumentException.class, () -> fpTensor.rightShift(fpTensor));

        assertThrows(
                IllegalArgumentException.class, () -> boolTensor.rightShiftUnsigned(boolTensor));
        assertThrows(IllegalArgumentException.class, () -> fpTensor.rightShiftUnsigned(fpTensor));
    }

    private static int normalizedShift(DataType dataType, long shiftCount) {
        if (dataType == DataType.I8) {
            return ((int) shiftCount) & 7;
        }
        if (dataType == DataType.I16) {
            return ((int) shiftCount) & 15;
        }
        if (dataType == DataType.I32) {
            return ((int) shiftCount) & 31;
        }
        return ((int) shiftCount) & 63;
    }

    private static long expectedLeft(long value, int shift, DataType dataType) {
        if (dataType == DataType.I8) {
            return (byte) ((byte) value << shift);
        }
        if (dataType == DataType.I16) {
            return (short) ((short) value << shift);
        }
        if (dataType == DataType.I32) {
            return (int) value << shift;
        }
        return value << shift;
    }

    private static long expectedRight(long value, int shift, DataType dataType) {
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

    private static long expectedRightUnsigned(long value, int shift, DataType dataType) {
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

    private static int readInt(Tensor tensor, long linearIndex) {
        return (int) TensorTestReads.readValue(tensor, linearIndex, DataType.I32);
    }

    private static long readIntegral(Tensor tensor, long linearIndex, DataType dataType) {
        return ((Number) TensorTestReads.readValue(tensor, linearIndex, dataType)).longValue();
    }
}
