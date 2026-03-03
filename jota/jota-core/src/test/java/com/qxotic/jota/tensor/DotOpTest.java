package com.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.testutil.RunOnAllAvailableBackends;
import com.qxotic.jota.testutil.TensorTestReads;
import java.util.List;
import org.junit.jupiter.api.Test;

@RunOnAllAvailableBackends
class DotOpTest {

    private static final List<DataType> FLOATING_TYPES =
            List.of(DataType.FP16, DataType.BF16, DataType.FP32, DataType.FP64);

    private static final List<DataType> INTEGRAL_TYPES =
            List.of(DataType.I8, DataType.I16, DataType.I32, DataType.I64);

    @Test
    void dotWorksForFloatVectors() {
        Tensor a = Tensor.of(new float[] {1f, 2f, 3f});
        Tensor b = Tensor.of(new float[] {4f, 5f, 6f});

        Tensor outTensor = a.dot(b);
        MemoryView<?> out = outTensor.materialize();
        assertEquals(Shape.scalar(), out.shape());
        assertEquals(DataType.FP32, out.dataType());
        assertEquals(32.0f, asFloat(outTensor), 1e-5f);
    }

    @Test
    void defaultDotSupportsAllFloatingTypes() {
        for (DataType type : FLOATING_TYPES) {
            Tensor a = Tensor.of(new long[] {1L, 2L, 3L}).cast(type);
            Tensor b = Tensor.of(new long[] {4L, 5L, 6L}).cast(type);
            MemoryView<?> out = a.dot(b).materialize();
            assertEquals(Shape.scalar(), out.shape());
            assertEquals(type, out.dataType());
        }
    }

    @Test
    void dotUsesAccumulatorTypeForMultiplyAndSum() {
        Tensor a = Tensor.of(new int[] {50_000, 50_000});
        Tensor b = Tensor.of(new int[] {50_000, 50_000});

        Tensor outTensor = a.dot(b, DataType.I64);
        MemoryView<?> out = outTensor.materialize();
        assertEquals(DataType.I64, out.dataType());
        assertEquals(5_000_000_000L, asLong(outTensor));
    }

    @Test
    void dotWithoutAccumulatorRejectsIntegralInputs() {
        Tensor a = Tensor.of(new int[] {1, 2, 3});
        Tensor b = Tensor.of(new int[] {4, 5, 6});
        assertThrows(IllegalArgumentException.class, () -> a.dot(b));
    }

    @Test
    void dotWithoutAccumulatorRejectsMixedFloatingAndIntegralInputs() {
        Tensor floats = Tensor.of(new float[] {1f, 2f, 3f});
        Tensor ints = Tensor.of(new int[] {1, 2, 3});
        assertThrows(IllegalArgumentException.class, () -> floats.dot(ints));
        assertThrows(IllegalArgumentException.class, () -> ints.dot(floats));
    }

    @Test
    void dotWithExplicitAccumulatorWorksForAllIntegralInputTypes() {
        for (DataType type : INTEGRAL_TYPES) {
            Tensor a = Tensor.of(new long[] {1L, 2L, 3L}).cast(type);
            Tensor b = Tensor.of(new long[] {4L, 5L, 6L}).cast(type);
            Tensor outTensor = a.dot(b, DataType.I64);
            MemoryView<?> out = outTensor.materialize();
            assertEquals(DataType.I64, out.dataType());
            assertEquals(32L, asLong(outTensor));
        }
    }

    @Test
    void dotAccumulatorTypeCanPromoteFloatPrecision() {
        Tensor a = Tensor.of(new float[] {1f, 2f, 3f});
        Tensor b = Tensor.of(new float[] {0.25f, 0.5f, 0.75f});

        Tensor outTensor = a.dot(b, DataType.FP64);
        MemoryView<?> out = outTensor.materialize();
        assertEquals(DataType.FP64, out.dataType());
        assertEquals(3.5d, asDouble(outTensor), 1e-10);
    }

    @Test
    void dotRejectsNarrowingAccumulatorType() {
        Tensor ints = Tensor.of(new int[] {1, 2, 3});
        assertThrows(IllegalArgumentException.class, () -> ints.dot(ints, DataType.I16));

        Tensor fp64 = Tensor.of(new double[] {1.0, 2.0, 3.0});
        assertThrows(IllegalArgumentException.class, () -> fp64.dot(fp64, DataType.FP32));
    }

    @Test
    void tracedAndEagerDotMatch() {
        Tensor a = Tensor.of(new float[] {2f, -1f, 0.5f});
        Tensor b = Tensor.of(new float[] {3f, 4f, 5f});

        Tensor eager = a.dot(b, DataType.FP64);
        Tensor traced = Tracer.trace(a, b, (x, y) -> x.dot(y, DataType.FP64));

        assertEquals(asDouble(eager), asDouble(traced), 1e-12);
    }

    @Test
    void dotRejectsBoolInputs() {
        Tensor bools = Tensor.of(new long[] {0L, 1L, 0L}).cast(DataType.BOOL);
        Tensor ints = Tensor.of(new int[] {1, 2, 3});

        assertThrows(IllegalArgumentException.class, () -> bools.dot(bools));
        assertThrows(IllegalArgumentException.class, () -> ints.dot(bools));
    }

    @Test
    void dotRejectsMismatchedDtypes() {
        Tensor ints = Tensor.of(new int[] {1, 2, 3});
        Tensor longs = Tensor.of(new long[] {1L, 2L, 3L});
        assertThrows(IllegalArgumentException.class, () -> ints.dot(longs));
    }

    @Test
    void dotRejectsRankMismatchAndLengthMismatch() {
        Tensor vector = Tensor.of(new int[] {1, 2, 3});
        Tensor matrix = Tensor.iota(6, DataType.I32).view(Shape.of(2, 3));
        Tensor shortVector = Tensor.of(new int[] {1, 2});

        assertThrows(IllegalArgumentException.class, () -> vector.dot(matrix));
        assertThrows(IllegalArgumentException.class, () -> matrix.dot(vector));
        assertThrows(IllegalArgumentException.class, () -> vector.dot(shortVector));
    }

    @Test
    void dotRejectsEmptyVectorsAndInvalidAccumulator() {
        Tensor empty = Tensor.of(new int[] {}).view(Shape.of(0));
        Tensor vector = Tensor.of(new int[] {1, 2, 3});

        assertThrows(IllegalArgumentException.class, () -> empty.dot(empty));
        assertThrows(IllegalArgumentException.class, () -> vector.dot(vector, DataType.BOOL));
    }

    private static float asFloat(Tensor tensor) {
        return (float) TensorTestReads.readValue(tensor, 0, DataType.FP32);
    }

    private static double asDouble(Tensor tensor) {
        return (double) TensorTestReads.readValue(tensor, 0, DataType.FP64);
    }

    private static long asLong(Tensor tensor) {
        return TensorTestReads.readLong(tensor, 0);
    }
}
