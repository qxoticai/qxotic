package com.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Device;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.testutil.ConfiguredTestDevice;
import com.qxotic.jota.testutil.RunOnAllAvailableBackends;
import com.qxotic.jota.testutil.TensorTestReads;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

@RunOnAllAvailableBackends
class RepeatInterleaveOpTest {

    @BeforeEach
    void assumeRepeatSupportedOnConfiguredDevice() {
        Assumptions.assumeTrue(
                ConfiguredTestDevice.resolve() == Device.PANAMA,
                "Repeat currently panama-only in runtime-agnostic lane");
    }

    @Test
    void repeatInterleaveRepeatsElementsAlongAxis() {
        Tensor input = Tensor.of(new float[] {1f, 2f, 3f, 4f}, Shape.of(2, 2));

        Tensor result = input.repeatInterleave(3, 1);
        MemoryView<?> output = result.materialize();

        assertEquals(Shape.of(2, 6), output.shape());
        float[] expected = {1f, 1f, 1f, 2f, 2f, 2f, 3f, 3f, 3f, 4f, 4f, 4f};
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], readFloat(result, i), 1e-4f);
        }
    }

    @Test
    void repeatInterleaveOnIotaOneDimRepeatsElements() {
        Tensor input = Tensor.iota(4, DataType.FP32);

        Tensor repeated = input.repeatInterleave(2, 0);
        MemoryView<?> output = repeated.materialize();

        assertEquals(Shape.of(8), output.shape());
        float[] expected = {0f, 0f, 1f, 1f, 2f, 2f, 3f, 3f};
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], readFloat(repeated, i), 1e-4f);
        }
    }

    @Test
    void repeatInterleaveRepeatsAlongFirstAxis() {
        Tensor input = Tensor.of(new float[] {1f, 2f, 3f, 4f}, Shape.of(2, 2));

        Tensor result = input.repeatInterleave(2, 0);
        MemoryView<?> output = result.materialize();

        assertEquals(Shape.of(4, 2), output.shape());
        float[] expected = {1f, 2f, 1f, 2f, 3f, 4f, 3f, 4f};
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], readFloat(result, i), 1e-4f);
        }
    }

    @Test
    void repeatInterleaveSupportsNegativeAxisWrap() {
        Tensor input = Tensor.of(new float[] {1f, 2f, 3f, 4f}, Shape.of(2, 2));

        Tensor result = input.repeatInterleave(2, -1);
        MemoryView<?> output = result.materialize();

        assertEquals(Shape.of(2, 4), output.shape());
        float[] expected = {1f, 1f, 2f, 2f, 3f, 3f, 4f, 4f};
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], readFloat(result, i), 1e-4f);
        }
    }

    @Test
    void repeatInterleaveIdentityReturnsSameTensor() {
        Tensor input = Tensor.iota(6, DataType.FP32).view(Shape.of(2, 3));

        Tensor result = input.repeatInterleave(1, 1);

        assertSame(input, result);
    }

    @Test
    void repeatInterleaveSupportsNestedModeLayouts() {
        Tensor input = Tensor.iota(8, DataType.FP32).add(1f).view(Shape.of(2, Shape.of(2, 2)));

        Tensor repeated = input.repeatInterleave(2, 1);
        MemoryView<?> output = repeated.materialize();

        assertEquals(Shape.of(2, 8), output.shape());
        float[] expected = {1f, 1f, 2f, 2f, 3f, 3f, 4f, 4f, 5f, 5f, 6f, 6f, 7f, 7f, 8f, 8f};
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], readFloat(repeated, i), 1e-4f);
        }
    }

    @Test
    void repeatInterleaveWorksOnNonContiguousInput() {
        Tensor base = Tensor.iota(12, DataType.FP32).view(Shape.of(3, 4));
        Tensor transposed = base.transpose(0, 1);

        Tensor interleaved = transposed.repeatInterleave(2, 1);
        MemoryView<?> interleavedView = interleaved.materialize();

        assertEquals(Shape.of(4, 6), interleavedView.shape());
    }

    @Test
    void repeatInterleaveValidatesRepeats() {
        Tensor input = Tensor.iota(6, DataType.FP32).view(Shape.of(2, 3));

        assertThrows(IllegalArgumentException.class, () -> input.repeatInterleave(0, 1));
        assertThrows(IllegalArgumentException.class, () -> input.repeatInterleave(-1, 1));
        assertThrows(IllegalArgumentException.class, () -> input.repeatInterleave(2, 2));
        assertThrows(IllegalArgumentException.class, () -> input.repeatInterleave(2, -3));
    }

    @Test
    void repeatInterleaveRejectsScalarAxis() {
        Tensor scalar = Tensor.scalar(1.0f);

        assertThrows(IllegalArgumentException.class, () -> scalar.repeatInterleave(2, 0));
    }

    @Test
    void repeatInterleaveWorksThroughTracing() {
        Tensor input = Tensor.of(new float[] {1f, 2f, 3f, 4f}, Shape.of(2, 2));

        Tensor repeated = Tracer.trace(input, t -> t.repeatInterleave(2, 1));
        MemoryView<?> output = repeated.materialize();

        assertEquals(Shape.of(2, 4), output.shape());
        float[] expected = {1f, 1f, 2f, 2f, 3f, 3f, 4f, 4f};
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], readFloat(repeated, i), 1e-4f);
        }
    }

    private static float readFloat(Tensor tensor, long linearIndex) {
        return TensorTestReads.readFloat(tensor, linearIndex);
    }
}
