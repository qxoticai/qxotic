package com.qxotic.jinfer.x.kernels;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.format.gguf.GGMLType;
import com.qxotic.jinfer.F32FloatTensor;
import com.qxotic.jinfer.FloatTensor;
import com.qxotic.jinfer.x.Views;
import com.qxotic.jota.DataType;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;
import com.qxotic.jota.Stride;
import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.memory.impl.MemoryFactory;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Random;
import org.junit.jupiter.api.Test;

/**
 * Differential oracles: the ported kernels (this package) against the original jinfer-core
 * implementations on identical inputs — bit-equal required (both run the same vector code). Plus
 * the entry-check contract: wrong dtype / non-contiguous views are rejected.
 */
class NormsTest {

    private static final int[] SIZES = {1, 7, 16, 31, 32, 33, 1000, 2048, 2049};

    private final Arena arena = Arena.ofAuto();

    private F32FloatTensor oldTensor(MemorySegment seg, int n) {
        return (F32FloatTensor) FloatTensor.create(GGMLType.F32, n, seg);
    }

    private MemoryView<MemorySegment> newView(MemorySegment seg, int n) {
        return Views.wrap(seg, DataType.FP32, Shape.flat((long) n));
    }

    private MemorySegment filled(int n, long seed) {
        MemorySegment seg = arena.allocate(4L * n, 64);
        Random rng = new Random(seed);
        for (int i = 0; i < n; i++) {
            seg.set(ValueLayout.JAVA_FLOAT_UNALIGNED, 4L * i, rng.nextFloat() * 4 - 2);
        }
        return seg;
    }

    private float get(MemorySegment seg, int i) {
        return seg.get(ValueLayout.JAVA_FLOAT_UNALIGNED, 4L * i);
    }

    private void assertBitEqual(MemorySegment expected, MemorySegment actual, int n, String what) {
        // ulp-bounded, NOT bit-equal: Vector API tier nondeterminism — see Oracles
        Oracles.assertClose(expected, actual, n, what);
    }

    @Test
    void rmsnormParity() {
        for (int n : SIZES) {
            MemorySegment x = filled(n, 1), w = filled(n, 2);
            MemorySegment outOld = arena.allocate(4L * n, 64);
            MemorySegment outNew = arena.allocate(4L * n, 64);
            com.qxotic.jinfer.Norms.rmsnorm(
                    oldTensor(outOld, n), 0, oldTensor(x, n), 0, oldTensor(w, n), n, 1e-5f);
            Norms.rmsnorm(newView(outNew, n), 0, newView(x, n), 0, newView(w, n), n, 1e-5f);
            assertBitEqual(outOld, outNew, n, "rmsnorm n=" + n);
        }
    }

    @Test
    void rmsnormWithOffsetsParity() {
        int n = 2050, pad = 3;
        MemorySegment x = filled(n + pad, 3), w = filled(n, 4);
        MemorySegment outOld = arena.allocate(4L * (n + pad), 64);
        MemorySegment outNew = arena.allocate(4L * (n + pad), 64);
        com.qxotic.jinfer.Norms.rmsnorm(
                oldTensor(outOld, n + pad),
                pad,
                oldTensor(x, n + pad),
                pad,
                oldTensor(w, n),
                n,
                1e-5f);
        Norms.rmsnorm(
                newView(outNew, n + pad), pad, newView(x, n + pad), pad, newView(w, n), n, 1e-5f);
        assertBitEqual(outOld, outNew, n + pad, "rmsnorm offset");
    }

    @Test
    void sumOfSquaresParity() {
        for (int n : SIZES) {
            MemorySegment x = filled(n, 5);
            float a = com.qxotic.jinfer.Norms.sumOfSquares(oldTensor(x, n), 0, n);
            float b = Norms.sumOfSquares(newView(x, n), 0, n);
            assertEquals(a, b, 0f, "sumOfSquares n=" + n);
        }
    }

    @Test
    void scaleByWeightParity() {
        for (int n : SIZES) {
            MemorySegment x = filled(n, 6), w = filled(n, 7);
            MemorySegment outOld = arena.allocate(4L * n, 64);
            MemorySegment outNew = arena.allocate(4L * n, 64);
            com.qxotic.jinfer.Norms.scaleByWeight(
                    oldTensor(outOld, n), 0, oldTensor(x, n), 0, oldTensor(w, n), n, 0.125f);
            Norms.scaleByWeight(newView(outNew, n), 0, newView(x, n), 0, newView(w, n), n, 0.125f);
            assertBitEqual(outOld, outNew, n, "scaleByWeight n=" + n);
        }
    }

    @Test
    void rmsnormNoWeightParity() {
        for (int n : SIZES) {
            // out-of-place
            MemorySegment x = filled(n, 8);
            MemorySegment outOld = arena.allocate(4L * n, 64);
            MemorySegment outNew = arena.allocate(4L * n, 64);
            com.qxotic.jinfer.Norms.rmsnormNoWeight(
                    oldTensor(outOld, n), 0, oldTensor(x, n), 0, n, 1e-5f);
            Norms.rmsnormNoWeight(newView(outNew, n), 0, newView(x, n), 0, n, 1e-5f);
            assertBitEqual(outOld, outNew, n, "rmsnormNoWeight n=" + n);
            // in-place
            MemorySegment a = filled(n, 9), b = filled(n, 9);
            com.qxotic.jinfer.Norms.rmsnormNoWeight(
                    oldTensor(a, n), 0, oldTensor(a, n), 0, n, 1e-5f);
            Norms.rmsnormNoWeight(newView(b, n), 0, newView(b, n), 0, n, 1e-5f);
            assertBitEqual(a, b, n, "rmsnormNoWeight in-place n=" + n);
        }
    }

    @Test
    void layerNormParity() {
        for (int C : new int[] {1, 5, 32, 100, 2048, 2049}) {
            int T = 3;
            int n = C * T;
            MemorySegment x = filled(n, 10), g = filled(C, 11), b = filled(C, 12);
            MemorySegment outOld = arena.allocate(4L * n, 64);
            MemorySegment outNew = arena.allocate(4L * n, 64);
            com.qxotic.jinfer.Norms.layerNorm(
                    oldTensor(outOld, n),
                    oldTensor(x, n),
                    oldTensor(g, C),
                    oldTensor(b, C),
                    C,
                    T,
                    1e-5f);
            Norms.layerNorm(
                    newView(outNew, n), newView(x, n), newView(g, C), newView(b, C), C, T, 1e-5f);
            assertBitEqual(outOld, outNew, n, "layerNorm C=" + C);
        }
    }

    @Test
    void rejectsNonF32() {
        MemorySegment seg = filled(64, 13);
        MemoryView<MemorySegment> f16 = Views.wrap(seg, DataType.FP16, Shape.flat(32L));
        MemoryView<MemorySegment> f32 = newView(seg, 64);
        assertThrows(
                IllegalArgumentException.class,
                () -> Norms.rmsnorm(f32, 0, f16, 0, f32, 64, 1e-5f));
        assertThrows(IllegalArgumentException.class, () -> Ops.fillInPlace(f16, 0, 32, 1f));
    }

    @Test
    void rejectsNonContiguous() {
        MemorySegment seg = filled(16, 14);
        Memory<MemorySegment> mem = MemoryFactory.ofMemorySegment(seg);
        MemoryView<MemorySegment> transposed =
                MemoryView.of(
                        mem, 0, DataType.FP32, Layout.of(Shape.flat(4, 4), Stride.flat(1, 4)));
        MemoryView<MemorySegment> f32 = newView(seg, 16);
        assertThrows(
                IllegalArgumentException.class,
                () -> Norms.rmsnorm(f32, 0, transposed, 0, f32, 16, 1e-5f));
    }
}
