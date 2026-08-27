package com.qxotic.jinfer.kernels;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.Views;
import com.qxotic.jota.memory.MemoryAllocators;
import java.lang.foreign.Arena;
import org.junit.jupiter.api.Test;

/** Small semantic contracts for edge cases. */
class OpsContractTest {

    @Test
    void fusedActivationMatchesItsScalarDefinitionAcrossTheVectorTail() {
        try (Arena arena = Arena.ofConfined()) {
            var memory = MemoryAllocators.ofArena(arena);
            float[] gate = new float[35];
            float[] up = new float[35];
            for (int i = 0; i < gate.length; i++) {
                gate[i] = (i - 17) / 5f;
                up[i] = (i % 7) - 3;
            }
            var actual = Views.fromFloatArray(memory, gate);
            var multiplier = Views.fromFloatArray(memory, up);

            Activations.siluMultiply(actual, 1, multiplier, 1, 33);

            float[] result = Views.toFloatArray(actual, "gate");
            assertEquals(gate[0], result[0]);
            assertEquals(gate[34], result[34]);
            for (int i = 1; i < 34; i++) {
                assertEquals(Activations.silu(gate[i]) * up[i], result[i], 1e-4f, "lane " + i);
            }
        }
    }

    @Test
    void softmaxIsNormalizedAndShiftInvariant() {
        try (Arena arena = Arena.ofConfined()) {
            var memory = MemoryAllocators.ofArena(arena);
            float[] values = new float[33];
            float[] shifted = new float[33];
            for (int i = 0; i < values.length; i++) {
                values[i] = (i % 11) - 8;
                shifted[i] = values[i] + 100;
            }
            var a = Views.fromFloatArray(memory, values);
            var b = Views.fromFloatArray(memory, shifted);

            Ops.softmaxInPlace(a, 0, 33);
            Ops.softmaxInPlace(b, 0, 33);

            float sum = 0;
            float[] av = Views.toFloatArray(a, "a");
            float[] bv = Views.toFloatArray(b, "b");
            for (int i = 0; i < av.length; i++) {
                sum += av[i];
                assertEquals(av[i], bv[i], 2e-6f, "lane " + i);
                assertTrue(av[i] >= 0);
            }
            assertEquals(1f, sum, 2e-6f);
        }
    }

    @Test
    void argmaxIsRelativeToItsWindowAndKeepsTheFirstTie() {
        try (Arena arena = Arena.ofConfined()) {
            var values =
                    Views.fromFloatArray(
                            MemoryAllocators.ofArena(arena), new float[] {0, 3, 1, 0, 9, 9, 0, 2});
            assertEquals(1, Ops.argmax(values, 0, 4));
            assertEquals(0, Ops.argmax(values, 4, 4));
            assertEquals(4, Ops.argmax(values, 0, 8));
        }
    }

    @Test
    void windowedMeanPoolOwnsItsOutput() {
        // 3x3 patches of width 2 merged 2x2: the output is a sum the pool must start itself, not
        // an accumulator it inherits from whatever the destination held
        try (Arena arena = Arena.ofConfined()) {
            var memory = MemoryAllocators.ofArena(arena);
            float[] patches = new float[9 * 2];
            for (int i = 0; i < patches.length; i++) patches[i] = i;
            var src = Views.fromFloatArray(memory, patches);
            var dst = Views.fromFloatArray(memory, new float[] {1e6f, -1e6f});

            Ops.windowedMeanPool(src, 3, 3, 2, 2, dst);

            // rows (0,0) (0,1) (1,0) (1,1) of the 3x3 grid: patch indices 0, 1, 3, 4
            float[] expected = {(0 + 2 + 6 + 8) / 4f, (1 + 3 + 7 + 9) / 4f};
            float[] actual = Views.toFloatArray(dst, "pooled");
            assertEquals(expected[0], actual[0]);
            assertEquals(expected[1], actual[1]);
        }
    }
}
