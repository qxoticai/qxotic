package com.qxotic.jinfer.x.kernels;

import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jinfer.x.Views;
import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import java.lang.foreign.Arena;
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
}
