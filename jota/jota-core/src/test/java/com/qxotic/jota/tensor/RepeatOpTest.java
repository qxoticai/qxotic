package com.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jota.DataType;
import com.qxotic.jota.DeviceType;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.testutil.ConfiguredTestDevice;
import com.qxotic.jota.testutil.RunOnAllAvailableBackends;
import com.qxotic.jota.testutil.TensorTestReads;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

@RunOnAllAvailableBackends
class RepeatOpTest {

    @BeforeEach
    void assumeRepeatSupportedOnConfiguredDevice() {
        Assumptions.assumeTrue(
                ConfiguredTestDevice.resolve().belongsTo(DeviceType.PANAMA),
                "Repeat currently panama-only in runtime-agnostic lane");
    }

    @Test
    void repeatTilesAlongAxes() {
        Tensor input = Tensor.of(new float[] {1f, 2f, 3f, 4f}, Shape.of(2, 2));

        Tensor result = input.repeat(2, 3);
        MemoryView<?> output = result.materialize();

        assertEquals(Shape.of(4, 6), output.shape());
        float[] expected = {
            1f, 2f, 1f, 2f, 1f, 2f,
            3f, 4f, 3f, 4f, 3f, 4f,
            1f, 2f, 1f, 2f, 1f, 2f,
            3f, 4f, 3f, 4f, 3f, 4f
        };
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], readFloat(result, i), 1e-4f);
        }
    }

    @Test
    void repeatOnIotaOneDimTilesSequence() {
        Tensor input = Tensor.iota(4, DataType.FP32);

        Tensor repeated = input.repeat(2);
        MemoryView<?> output = repeated.materialize();

        assertEquals(Shape.of(8), output.shape());
        float[] expected = {0f, 1f, 2f, 3f, 0f, 1f, 2f, 3f};
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], readFloat(repeated, i), 1e-4f);
        }
    }

    @Test
    void repeatIdentityReturnsSameShapeAndValues() {
        Tensor input = Tensor.iota(6, DataType.FP32).view(Shape.of(2, 3));

        Tensor result = input.repeat(1, 1);
        assertSame(input, result);
        MemoryView<?> output = result.materialize();

        assertEquals(Shape.of(2, 3), output.shape());
        for (int i = 0; i < 6; i++) {
            assertEquals((float) i, readFloat(result, i), 1e-4f);
        }
    }

    @Test
    void repeatSupportsHigherRankShapes() {
        Tensor input = Tensor.iota(12, DataType.FP32).view(Shape.of(2, 2, 3));

        Tensor repeated = input.repeat(2, 1, 2);
        MemoryView<?> output = repeated.materialize();

        assertEquals(Shape.of(4, 2, 6), output.shape());
        assertEquals(0f, readFloat(repeated, 0), 1e-4f);
        assertEquals(1f, readFloat(repeated, 1), 1e-4f);
        assertEquals(0f, readFloat(repeated, 3), 1e-4f);
    }

    @Test
    void repeatSupportsNestedModeLayouts() {
        Tensor input = Tensor.iota(8, DataType.FP32).add(1f).view(Shape.of(2, Shape.of(2, 2)));

        Tensor repeated = input.repeat(2, 2);
        MemoryView<?> output = repeated.materialize();

        assertEquals(Shape.of(4, Shape.of(4, 2)), output.shape());
        float[] expected = {
            1f, 2f, 3f, 4f, 1f, 2f, 3f, 4f,
            5f, 6f, 7f, 8f, 5f, 6f, 7f, 8f,
            1f, 2f, 3f, 4f, 1f, 2f, 3f, 4f,
            5f, 6f, 7f, 8f, 5f, 6f, 7f, 8f
        };
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], readFloat(repeated, i), 1e-4f);
        }
    }

    @Test
    void repeatWorksOnNonContiguousInput() {
        Tensor base = Tensor.iota(12, DataType.FP32).view(Shape.of(3, 4));
        Tensor transposed = base.transpose(0, 1);

        Tensor tiled = transposed.repeat(2, 1);
        MemoryView<?> tiledView = tiled.materialize();

        assertEquals(Shape.of(8, 3), tiledView.shape());
    }

    @Test
    void repeatValidatesInputsStrictly() {
        Tensor input = Tensor.iota(6, DataType.FP32).view(Shape.of(2, 3));

        assertThrows(IllegalArgumentException.class, () -> input.repeat());
        assertThrows(IllegalArgumentException.class, () -> input.repeat(2));
        assertThrows(IllegalArgumentException.class, () -> input.repeat(2, 3, 4));
        assertThrows(IllegalArgumentException.class, () -> input.repeat(0, 1));
        assertThrows(IllegalArgumentException.class, () -> input.repeat(1, -1));
    }

    @Test
    void repeatRejectsNestedFlatRankStyleArguments() {
        Tensor nested = Tensor.iota(8, DataType.FP32).view(Shape.of(2, Shape.of(2, 2)));

        assertThrows(IllegalArgumentException.class, () -> nested.repeat(2, 2, 2));
    }

    @Test
    void repeatSupportsScalarIdentityAndRejectsExplicitFactors() {
        Tensor scalar = Tensor.scalar(3.0f);

        assertSame(scalar, scalar.repeat());
        assertThrows(IllegalArgumentException.class, () -> scalar.repeat(1));
        assertThrows(IllegalArgumentException.class, () -> scalar.repeat(-1));
    }

    @Test
    void repeatWorksThroughTracing() {
        Tensor input = Tensor.of(new float[] {1f, 2f, 3f, 4f}, Shape.of(2, 2));

        Tensor repeated = Tracer.trace(input, t -> t.repeat(2, 2));
        MemoryView<?> output = repeated.materialize();

        assertEquals(Shape.of(4, 4), output.shape());
        float[] expected = {1f, 2f, 1f, 2f, 3f, 4f, 3f, 4f, 1f, 2f, 1f, 2f, 3f, 4f, 3f, 4f};
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], readFloat(repeated, i), 1e-4f);
        }
    }

    @Test
    void repeatPreservesIntegralType() {
        Tensor input = Tensor.of(new int[] {1, 2, 3, 4}, Shape.of(2, 2));

        Tensor repeated = input.repeat(1, 2);
        MemoryView<?> output = repeated.materialize();

        assertEquals(DataType.I32, output.dataType());
        int[] expected = {1, 2, 1, 2, 3, 4, 3, 4};
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], readInt(repeated, i));
        }
    }

    private static float readFloat(Tensor tensor, long linearIndex) {
        return TensorTestReads.readFloat(tensor, linearIndex);
    }

    private static int readInt(Tensor tensor, long linearIndex) {
        return ((Number) TensorTestReads.readValue(tensor, linearIndex, DataType.I32)).intValue();
    }
}
