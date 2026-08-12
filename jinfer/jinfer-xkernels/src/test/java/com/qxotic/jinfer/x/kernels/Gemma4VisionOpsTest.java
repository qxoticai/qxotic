package com.qxotic.jinfer.x.kernels;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.junit.jupiter.api.Test;

class Gemma4VisionOpsTest {

    private final Arena arena = Arena.ofAuto();

    @Test
    void clampsFp32SpanInPlace() {
        float[] values = {-9f, -2f, -1.25f, -0f, 0.5f, 3f, 8f, 99f};
        MemorySegment segment = f32(values);

        Ops.clampInPlace(view(segment, values.length), 1, values.length - 2, -1.25f, 3f);

        for (int i = 0; i < values.length; i++) {
            float expected = values[i];
            if (i >= 1 && i < values.length - 1) {
                expected = expected < -1.25f ? -1.25f : expected > 3f ? 3f : expected;
            }
            assertEquals(expected, get(segment, i), 0f, "lane " + i);
        }
    }

    @Test
    void broadcastsBiasAcrossRows() {
        int rows = 3, cols = 17, offset = 2, biasOffset = 1;
        float[] values = new float[offset + rows * cols + 1];
        float[] bias = new float[biasOffset + cols];
        for (int i = 0; i < values.length; i++) values[i] = i * 0.25f - 3f;
        for (int i = 0; i < bias.length; i++) bias[i] = 2f - i * 0.125f;
        MemorySegment actual = f32(values);

        Ops.addRowBiasInPlace(
                view(actual, values.length),
                offset,
                view(f32(bias), bias.length),
                biasOffset,
                rows,
                cols);

        for (int i = 0; i < values.length; i++) {
            float expected = values[i];
            if (i >= offset && i < offset + rows * cols) {
                expected += bias[biasOffset + (i - offset) % cols];
            }
            assertEquals(expected, get(actual, i), 0f, "lane " + i);
        }
    }

    @Test
    void quickGeluMultiplyMatchesScalarFormulaExactly() {
        float[] gate = {-100f, -7f, -1f, -0f, 0.25f, 1f, 7f, 100f, 42f};
        float[] up = {0.5f, -2f, 3f, 4f, -5f, 6f, 0.125f, -0.25f, 99f};
        MemorySegment actual = f32(gate);

        Activations.quickGeluMultiply(
                view(actual, gate.length), 0, view(f32(up), up.length), 0, gate.length - 1);

        for (int i = 0; i < gate.length - 1; i++) {
            float activated = gate[i] / (1f + (float) Math.exp(-1.702f * gate[i]));
            assertEquals(activated * up[i], get(actual, i), 0f, "lane " + i);
        }
        assertEquals(gate[gate.length - 1], get(actual, gate.length - 1), 0f);
    }

    private MemorySegment f32(float[] values) {
        MemorySegment segment = arena.allocate(values.length * Float.BYTES, 64);
        for (int i = 0; i < values.length; i++) {
            segment.set(ValueLayout.JAVA_FLOAT_UNALIGNED, (long) i * Float.BYTES, values[i]);
        }
        return segment;
    }

    private static float get(MemorySegment segment, int index) {
        return segment.get(ValueLayout.JAVA_FLOAT_UNALIGNED, (long) index * Float.BYTES);
    }

    private static MemoryView<MemorySegment> view(MemorySegment segment, int size) {
        return Oracles.f32View(segment, size);
    }
}
