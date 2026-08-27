package com.qxotic.jinfer.kernels;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.Views;
import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryAllocators;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.junit.jupiter.api.Test;

class MatMulTest {

    @Test
    void rejectsNonFp32Activations() {
        try (Arena arena = Arena.ofConfined()) {
            var weights = Oracles.q8(arena, 4, 64, 7);
            var input = Oracles.f32(arena, 64, 8);
            var output = Oracles.f32(arena, 4, 9);
            var f16 = Views.wrap(input, DataType.FP16, Shape.flat(32));

            assertThrows(
                    IllegalArgumentException.class,
                    () ->
                            MatMul.mm(
                                    Oracles.q8View(weights, 4L * 64),
                                    0,
                                    64,
                                    f16,
                                    0,
                                    64,
                                    Oracles.f32View(output, 4),
                                    0,
                                    4,
                                    4,
                                    1,
                                    64));
        }
    }

    @Test
    void f16WeightsKeepTheirSubnormals() {
        // a half below 2^-14 is subnormal: the vector body, the scalar tail and the scalar path
        // must all see its value, not zero
        int k = 67; // vector body plus a scalar tail
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment w = arena.allocate(k * 2L);
            float expected = 0f;
            for (int i = 0; i < k; i++) {
                float value = i % 3 == 0 ? 3e-5f : 0.25f;
                short bits = Float.floatToFloat16(value);
                w.set(ValueLayout.JAVA_SHORT, i * 2L, bits);
                expected += Float.float16ToFloat(bits);
            }
            float[] ones = new float[k];
            java.util.Arrays.fill(ones, 1f);
            var memory = MemoryAllocators.ofArena(arena);
            var x = Views.fromFloatArray(memory, ones);
            var out = Views.allocateF32(memory, 1);

            MatMul.gemv(Views.wrap(w, DataType.FP16, Shape.flat(k)), x, out, 1, k);

            assertEquals(
                    expected,
                    Views.getFloat(out, 0, "out"),
                    1e-5f); // lane order, not a flush (7e-4)
        }
    }

    @Test
    void aliasingIsJudgedByAddressNotByViewIdentity() {
        // two Memory objects over one segment: the shaped contract must still refuse in-place
        int m = 4, k = 8, n = 2;
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment shared = arena.allocate(n * k * 4L);
            var a = Views.wrap(shared, DataType.FP32, Shape.flat(n, k));
            var c = Views.wrap(shared, DataType.FP32, Shape.flat(n, k));
            var w = Views.wrap(arena.allocate(m * k * 4L), DataType.FP32, Shape.flat(m, k));
            IllegalArgumentException e =
                    assertThrows(IllegalArgumentException.class, () -> MatMul.gemm(w, a, c, n));
            assertTrue(e.getMessage().contains("alias"), e.getMessage());
        }
    }
}
