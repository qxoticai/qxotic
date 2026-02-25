package com.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Indexing;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryAccess;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.memory.impl.DomainFactory;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

class RepeatOpsTest {

    private static MemoryDomain<MemorySegment> domain;

    @BeforeAll
    static void setUpDomain() {
        domain = DomainFactory.ofMemorySegment();
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
            assertEquals(expected[i], readFloat(output, i), 1e-4f);
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
            assertEquals(expected[i], readFloat(output, i), 1e-4f);
        }
    }

    @Test
    void repeatInterleaveRepeatsElementsAlongAxis() {
        Tensor input = Tensor.of(new float[] {1f, 2f, 3f, 4f}, Shape.of(2, 2));

        Tensor result = input.repeatInterleave(3, 1);
        MemoryView<?> output = result.materialize();

        assertEquals(Shape.of(2, 6), output.shape());
        float[] expected = {1f, 1f, 1f, 2f, 2f, 2f, 3f, 3f, 3f, 4f, 4f, 4f};
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], readFloat(output, i), 1e-4f);
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
            assertEquals(expected[i], readFloat(output, i), 1e-4f);
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
            assertEquals(expected[i], readFloat(output, i), 1e-4f);
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
            assertEquals(expected[i], readFloat(output, i), 1e-4f);
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
            assertEquals((float) i, readFloat(output, i), 1e-4f);
        }
    }

    @Test
    void repeatInterleaveIdentityReturnsSameTensor() {
        Tensor input = Tensor.iota(6, DataType.FP32).view(Shape.of(2, 3));

        Tensor result = input.repeatInterleave(1, 1);

        assertSame(input, result);
    }

    @Test
    void repeatSupportsHigherRankShapes() {
        Tensor input = Tensor.iota(12, DataType.FP32).view(Shape.of(2, 2, 3));

        Tensor repeated = input.repeat(2, 1, 2);
        MemoryView<?> output = repeated.materialize();

        assertEquals(Shape.of(4, 2, 6), output.shape());
        assertEquals(0f, readFloat(output, 0), 1e-4f);
        assertEquals(1f, readFloat(output, 1), 1e-4f);
        assertEquals(0f, readFloat(output, 3), 1e-4f);
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
            assertEquals(expected[i], readFloat(output, i), 1e-4f);
        }
    }

    @Test
    void repeatInterleaveSupportsNestedModeLayouts() {
        Tensor input = Tensor.iota(8, DataType.FP32).add(1f).view(Shape.of(2, Shape.of(2, 2)));

        Tensor repeated = input.repeatInterleave(2, 1);
        MemoryView<?> output = repeated.materialize();

        assertEquals(Shape.of(2, 8), output.shape());
        float[] expected = {1f, 1f, 2f, 2f, 3f, 3f, 4f, 4f, 5f, 5f, 6f, 6f, 7f, 7f, 8f, 8f};
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], readFloat(output, i), 1e-4f);
        }
    }

    @Test
    void repeatAndRepeatInterleaveWorkOnNonContiguousInput() {
        Tensor base = Tensor.iota(12, DataType.FP32).view(Shape.of(3, 4));
        Tensor transposed = base.transpose(0, 1);

        Tensor tiled = transposed.repeat(2, 1);
        Tensor interleaved = transposed.repeatInterleave(2, 1);

        MemoryView<?> tiledView = tiled.materialize();
        MemoryView<?> interleavedView = interleaved.materialize();

        assertEquals(Shape.of(8, 3), tiledView.shape());
        assertEquals(Shape.of(4, 6), interleavedView.shape());
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

        // Nested rank is 2 (modes), even though flatRank is 3.
        assertThrows(IllegalArgumentException.class, () -> nested.repeat(2, 2, 2));
    }

    @Test
    void repeatRejectsScalarArguments() {
        Tensor scalar = Tensor.scalar(3.0f);

        assertThrows(IllegalArgumentException.class, () -> scalar.repeat());
        assertThrows(IllegalArgumentException.class, () -> scalar.repeat(1));
        assertThrows(IllegalArgumentException.class, () -> scalar.repeat(-1));
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
    void repeatWorksThroughTracing() {
        Tensor input = Tensor.of(new float[] {1f, 2f, 3f, 4f}, Shape.of(2, 2));

        Tensor repeated = Tracer.trace(input, t -> t.repeat(2, 2));
        MemoryView<?> output = repeated.materialize();

        assertEquals(Shape.of(4, 4), output.shape());
        float[] expected = {1f, 2f, 1f, 2f, 3f, 4f, 3f, 4f, 1f, 2f, 1f, 2f, 3f, 4f, 3f, 4f};
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], readFloat(output, i), 1e-4f);
        }
    }

    @Test
    void repeatInterleaveWorksThroughTracing() {
        Tensor input = Tensor.of(new float[] {1f, 2f, 3f, 4f}, Shape.of(2, 2));

        Tensor repeated = Tracer.trace(input, t -> t.repeatInterleave(2, 1));
        MemoryView<?> output = repeated.materialize();

        assertEquals(Shape.of(2, 4), output.shape());
        float[] expected = {1f, 1f, 2f, 2f, 3f, 3f, 4f, 4f};
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], readFloat(output, i), 1e-4f);
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
            assertEquals(expected[i], readInt(output, i));
        }
    }

    private static float readFloat(MemoryView<?> view, long linearIndex) {
        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment> typedView = (MemoryView<MemorySegment>) view;
        long offset = Indexing.linearToOffset(typedView, linearIndex);
        MemoryAccess<MemorySegment> access = domain.directAccess();
        return access.readFloat(typedView.memory(), offset);
    }

    private static int readInt(MemoryView<?> view, long linearIndex) {
        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment> typedView = (MemoryView<MemorySegment>) view;
        long offset = Indexing.linearToOffset(typedView, linearIndex);
        MemoryAccess<MemorySegment> access = domain.directAccess();
        return access.readInt(typedView.memory(), offset);
    }
}
