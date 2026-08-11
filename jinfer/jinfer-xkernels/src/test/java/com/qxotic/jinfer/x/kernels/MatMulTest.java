package com.qxotic.jinfer.x.kernels;

import static com.qxotic.jinfer.x.kernels.Oracles.assertClose;
import static com.qxotic.jinfer.x.kernels.Oracles.f32;
import static com.qxotic.jinfer.x.kernels.Oracles.f32View;
import static com.qxotic.jinfer.x.kernels.Oracles.oldF32;
import static com.qxotic.jinfer.x.kernels.Oracles.oldQ8;
import static com.qxotic.jinfer.x.kernels.Oracles.q8;
import static com.qxotic.jinfer.x.kernels.Oracles.q8View;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jinfer.F32FloatTensor;
import com.qxotic.jinfer.FloatTensor;
import com.qxotic.jinfer.x.Views;
import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;

/**
 * Differential oracles for {@link MatMul}: old {@code FloatTensor.matmul/gemm} (routed to the Java
 * floor — JAM backends are excluded from the test classpath) against {@code MatMul.mm} on identical
 * buffers. Ulp-bounded, NOT bit-equal — see {@link Oracles}.
 */
class MatMulTest {

    private final Arena arena = Arena.ofAuto();

    private void gemvParity(int m, int k, boolean useQ8) {
        MemorySegment w = useQ8 ? q8(arena, m, k, 1) : f32(arena, m * k, 1);
        MemorySegment x = f32(arena, k, 2);
        MemorySegment outOld = arena.allocate(4L * m, 64);
        MemorySegment outNew = arena.allocate(4L * m, 64);
        FloatTensor wOld = useQ8 ? oldQ8(w, (long) m * k) : oldF32(w, (long) m * k);
        MemoryView<MemorySegment> wNew = useQ8 ? q8View(w, (long) m * k) : f32View(w, (long) m * k);
        wOld.matmul(oldF32(x, k), oldF32(outOld, m), m, k);
        MatMul.mm(wNew, 0, k, f32View(x, k), 0, k, f32View(outNew, m), 0, m, m, 1, k);
        assertClose(
                outOld, outNew, m, "gemv m=" + m + " k=" + k + " q8=" + useQ8, Oracles.DOT_ABS_TOL);
    }

    @Test
    void gemvQ8Parity() {
        gemvParity(32, 64, true); // tiny: serial path
        gemvParity(128, 96, true);
        gemvParity(2048, 2048, true); // parallel path
        gemvParity(2048, 4096, true); // > TINY_MATVEC_ELEMS
    }

    @Test
    void gemvF32Parity() {
        gemvParity(32, 64, false);
        gemvParity(2048, 2048, false);
    }

    private void gemmParity(int m, int n, int k, boolean useQ8) {
        MemorySegment w = useQ8 ? q8(arena, m, k, 3) : f32(arena, m * k, 3);
        MemorySegment a = f32(arena, n * k, 4);
        MemorySegment outOld = arena.allocate(4L * n * m, 64);
        MemorySegment outNew = arena.allocate(4L * n * m, 64);
        FloatTensor wOld = useQ8 ? oldQ8(w, (long) m * k) : oldF32(w, (long) m * k);
        MemoryView<MemorySegment> wNew = useQ8 ? q8View(w, (long) m * k) : f32View(w, (long) m * k);
        wOld.gemm(oldF32(a, (long) n * k), k, oldF32(outOld, (long) n * m), m, n, m, k);
        MatMul.mm(
                wNew,
                0,
                k,
                f32View(a, (long) n * k),
                0,
                k,
                f32View(outNew, (long) n * m),
                0,
                m,
                m,
                n,
                k);
        assertClose(
                outOld,
                outNew,
                n * m,
                "gemm m=" + m + " n=" + n + " k=" + k + " q8=" + useQ8,
                Oracles.DOT_ABS_TOL);
    }

    @Test
    void gemmQ8Parity() {
        gemmParity(64, 2, 64, true);
        gemmParity(128, 5, 2048, true);
        gemmParity(512, 3, 2048, true);
    }

    @Test
    void gemmF32Parity() {
        gemmParity(64, 2, 64, false);
        gemmParity(128, 4, 512, false);
    }

    @Test
    void gemvInPlaceParity() {
        // a == c: the temp-buffer path on both sides. The OLD side must pass the SAME tensor
        // object for a and c — aliasing through two objects is outside the old contract.
        int m = 256, k = 256;
        MemorySegment w = q8(arena, m, k, 5);
        MemorySegment acOld = f32(arena, k, 6);
        MemorySegment acNew = f32(arena, k, 6);
        FloatTensor wOld = oldQ8(w, (long) m * k);
        F32FloatTensor ac = oldF32(acOld, k);
        wOld.matmul(ac, ac, m, k);
        MemoryView<MemorySegment> acv = f32View(acNew, k);
        MatMul.mm(q8View(w, (long) m * k), 0, k, acv, 0, k, acv, 0, m, m, 1, k);
        assertClose(acOld, acNew, m, "gemv in-place", Oracles.DOT_ABS_TOL);
    }

    @Test
    void rejectsBadOperands() {
        MemorySegment w = q8(arena, 4, 64, 7);
        MemorySegment x = f32(arena, 64, 8);
        MemorySegment out = f32(arena, 4, 9);
        MemoryView<MemorySegment> wv = q8View(w, 4L * 64);
        MemoryView<MemorySegment> f16 = Views.wrap(x, DataType.FP16, Shape.flat(32L));
        // FP16 activation: rejected by the FP32 entry check
        assertThrows(
                IllegalArgumentException.class,
                () -> MatMul.mm(wv, 0, 64, f16, 0, 64, f32View(out, 4), 0, 4, 4, 1, 64));
        // FP16 weights: unsupported weight dtype
        assertThrows(
                UnsupportedOperationException.class,
                () ->
                        MatMul.mm(
                                f16,
                                0,
                                64,
                                f32View(x, 64),
                                0,
                                64,
                                f32View(out, 4),
                                0,
                                4,
                                4,
                                1,
                                64));
    }
}
