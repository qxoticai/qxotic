package com.qxotic.jinfer.kernels;

import static com.qxotic.jinfer.kernels.Oracles.assertClose;
import static com.qxotic.jinfer.kernels.Oracles.f32;
import static com.qxotic.jinfer.kernels.Oracles.f32View;
import static com.qxotic.jinfer.kernels.Oracles.q8;
import static com.qxotic.jinfer.kernels.Oracles.q8View;

import com.qxotic.jinfer.Views;
import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;

/**
 * The shaped gemm/gemv entry points against the flat low-level gemm, with a block-quantized weight:
 * this pins the physical-vs-logical contract - {@code gemm} reads {@code m} from the weight's first
 * axis and reconstructs {@code k} by unfolding the blocked innermost axis, which must agree with
 * the flat path's explicit {@code (m, k)}.
 */
class ShapedMatMulParityTest {

    @Test
    void shapedGemvMatchesFlatGemvForQuantizedWeight() {
        int m = 64, k = 256;
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment w = q8(arena, m, k, 11);
            MemorySegment a = f32(arena, k, 22);
            MemorySegment flatOut = arena.allocate(4L * m, 64);
            MemorySegment shapedOut = arena.allocate(4L * m, 64);

            // Flat low-level path: weight is a 1D physical [m*k/32] view, A/C are flat FP32.
            MatMul.gemv(q8View(w, (long) m * k), f32View(a, k), f32View(flatOut, m), m, k);

            // Shaped path: weight is a 2D physical [m, k/32] view, A/C are 2D FP32 [1, inner].
            MemoryView<MemorySegment> w2 = Views.wrap(w, DataType.Q8_0, Shape.flat(m, k / 32));
            MemoryView<MemorySegment> a2 = Views.wrap(a, DataType.FP32, Shape.flat(1, k));
            MemoryView<MemorySegment> c2 = Views.wrap(shapedOut, DataType.FP32, Shape.flat(1, m));
            MatMul.gemv(w2, a2, c2);

            assertClose(flatOut, shapedOut, m, "shaped gemv vs flat gemv (Q8_0)");
        }
    }
}
