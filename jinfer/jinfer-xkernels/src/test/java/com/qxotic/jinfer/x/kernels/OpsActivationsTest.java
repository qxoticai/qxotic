package com.qxotic.jinfer.x.kernels;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.format.gguf.GGMLType;
import com.qxotic.jinfer.F32FloatTensor;
import com.qxotic.jinfer.FloatTensor;
import com.qxotic.jinfer.x.Views;
import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Random;
import org.junit.jupiter.api.Test;

/**
 * Differential oracles for {@link Ops} and {@link Activations} against the jinfer-core originals:
 * bit-equal on identical inputs (same vector code both sides), tail sizes included.
 */
class OpsActivationsTest {

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

    private void assertBitEqual(MemorySegment expected, MemorySegment actual, int n, String what) {
        // ulp-bounded, NOT bit-equal: Vector API tier nondeterminism — see Oracles
        Oracles.assertClose(expected, actual, n, what);
    }

    @Test
    void fillParity() {
        for (int n : SIZES) {
            MemorySegment a = arena.allocate(4L * n, 64), b = arena.allocate(4L * n, 64);
            oldTensor(a, n).fillInPlace(0, n, 3.25f);
            Ops.fillInPlace(newView(b, n), 0, n, 3.25f);
            assertBitEqual(a, b, n, "fill n=" + n);
        }
    }

    @Test
    void divideParity() {
        for (int n : SIZES) {
            MemorySegment a = filled(n, 1), b = filled(n, 1);
            oldTensor(a, n).divideInPlace(0, n, 7f);
            Ops.divideInPlace(newView(b, n), 0, n, 7f);
            assertBitEqual(a, b, n, "divide n=" + n);
        }
    }

    @Test
    void addParity() {
        for (int n : SIZES) {
            MemorySegment a = filled(n, 2), b = filled(n, 2), y = filled(n, 3);
            oldTensor(a, n).addInPlace(0, oldTensor(y, n), 0, n);
            Ops.addInPlace(newView(b, n), 0, newView(y, n), 0, n);
            assertBitEqual(a, b, n, "add n=" + n);
        }
    }

    @Test
    void addScaledParity() {
        for (int n : SIZES) {
            MemorySegment x1 = filled(n, 4),
                    x2 = filled(n, 4),
                    b1 = filled(n, 5),
                    b2 = filled(n, 5);
            FloatTensor.addScaled(oldTensor(x1, n), oldTensor(b1, n), n, 0.375f);
            Ops.addScaled(newView(x2, n), newView(b2, n), n, 0.375f);
            assertBitEqual(x1, x2, n, "addScaled x n=" + n);
            assertBitEqual(b1, b2, n, "addScaled xb n=" + n);
        }
    }

    @Test
    void addScaledIntoParity() {
        for (int n : SIZES) {
            MemorySegment base = filled(n + 2, 6), add = filled(n, 7);
            MemorySegment o1 = arena.allocate(4L * n, 64), o2 = arena.allocate(4L * n, 64);
            FloatTensor.addScaledInto(
                    oldTensor(o1, n), oldTensor(base, n + 2), 2, oldTensor(add, n), n, -0.5f);
            Ops.addScaledInto(newView(o2, n), newView(base, n + 2), 2, newView(add, n), n, -0.5f);
            assertBitEqual(o1, o2, n, "addScaledInto n=" + n);
        }
    }

    @Test
    void siluParity() {
        for (int n : SIZES) {
            MemorySegment a = filled(n, 8), b = filled(n, 8);
            oldTensor(a, n).siluInPlace(0, n);
            Ops.siluInPlace(newView(b, n), 0, n);
            assertBitEqual(a, b, n, "silu n=" + n);
        }
    }

    @Test
    void siluMultiplyParity() {
        for (int n : SIZES) {
            MemorySegment g1 = filled(n, 9), g2 = filled(n, 9), u = filled(n, 10);
            com.qxotic.jinfer.Activations.siluMultiply(oldTensor(g1, n), 0, oldTensor(u, n), 0, n);
            Activations.siluMultiply(newView(g2, n), 0, newView(u, n), 0, n);
            assertBitEqual(g1, g2, n, "siluMultiply n=" + n);
        }
    }

    @Test
    void reluSqrParity() {
        for (int n : SIZES) {
            MemorySegment a = filled(n, 11), b = filled(n, 11);
            com.qxotic.jinfer.Activations.reluSqr(oldTensor(a, n), 0, n);
            Activations.reluSqr(newView(b, n), 0, n);
            assertBitEqual(a, b, n, "reluSqr n=" + n);
        }
    }

    @Test
    void geluMultiplyParity() {
        for (int n : SIZES) {
            MemorySegment g1 = filled(n, 12), g2 = filled(n, 12), u = filled(n, 13);
            com.qxotic.jinfer.Activations.geluMultiply(oldTensor(g1, n), 0, oldTensor(u, n), 0, n);
            Activations.geluMultiply(newView(g2, n), 0, newView(u, n), 0, n);
            assertBitEqual(g1, g2, n, "geluMultiply n=" + n);
        }
    }

    @Test
    void clampedSwigluMultiplyParity() {
        for (int n : SIZES) {
            MemorySegment g1 = filled(n, 14), g2 = filled(n, 14), u = filled(n, 15);
            com.qxotic.jinfer.Activations.clampedSwigluMultiply(
                    oldTensor(g1, n), 0, oldTensor(u, n), 0, n);
            Activations.clampedSwigluMultiply(newView(g2, n), 0, newView(u, n), 0, n);
            assertBitEqual(g1, g2, n, "clampedSwigluMultiply n=" + n);
        }
    }

    @Test
    void tanhSigmoidGateParity() {
        for (int n : SIZES) {
            MemorySegment f = filled(n, 16), g = filled(n, 17);
            MemorySegment o1 = arena.allocate(4L * n, 64), o2 = arena.allocate(4L * n, 64);
            com.qxotic.jinfer.Activations.tanhSigmoidGate(
                    oldTensor(o1, n), 0, oldTensor(f, n), 0, oldTensor(g, n), 0, n);
            Activations.tanhSigmoidGate(newView(o2, n), 0, newView(f, n), 0, newView(g, n), 0, n);
            assertBitEqual(o1, o2, n, "tanhSigmoidGate n=" + n);
        }
    }

    @Test
    void softcapParity() {
        for (int n : SIZES) {
            MemorySegment a = filled(n, 18), b = filled(n, 18);
            com.qxotic.jinfer.Activations.softcap(oldTensor(a, n), 0, n, 30f);
            Activations.softcap(newView(b, n), 0, n, 30f);
            assertBitEqual(a, b, n, "softcap n=" + n);
        }
    }

    @Test
    void scalarFormsUnchanged() {
        for (float v : new float[] {-30f, -1.5f, 0f, 0.7f, 25f}) {
            assertEquals(com.qxotic.jinfer.Activations.silu(v), Activations.silu(v), 0f);
            assertEquals(com.qxotic.jinfer.Activations.gelu(v), Activations.gelu(v), 0f);
            assertEquals(com.qxotic.jinfer.Activations.softplus(v), Activations.softplus(v), 0f);
            assertEquals(
                    com.qxotic.jinfer.Activations.clampedSwiglu(v, v * 2),
                    Activations.clampedSwiglu(v, v * 2),
                    0f);
        }
    }

    @Test
    void argmaxParity() {
        for (int n : SIZES) {
            MemorySegment a = filled(n, 20 + n);
            assertEquals(
                    oldTensor(a, n).argmax(0, n), Ops.argmax(newView(a, n), 0, n), "argmax n=" + n);
        }
        // ties keep the FIRST max; negative-only spans work
        MemorySegment neg = arena.allocate(4L * 5, 64);
        for (int i = 0; i < 5; i++) neg.set(ValueLayout.JAVA_FLOAT_UNALIGNED, 4L * i, -5f + i);
        assertEquals(oldTensor(neg, 5).argmax(0, 5), Ops.argmax(newView(neg, 5), 0, 5));
    }

    @Test
    void saxpyParity() {
        for (int n : SIZES) {
            MemorySegment a = filled(n, 30), b = filled(n, 30);
            MemorySegment y = filled(n, 31);
            oldTensor(a, n).saxpyInPlace(0, oldTensor(y, n), 0, n, 0.375f);
            Ops.saxpyInPlace(newView(b, n), 0, newView(y, n), 0, n, 0.375f);
            assertBitEqual(a, b, n, "saxpy n=" + n);
        }
    }

    @Test
    void softmaxParity() {
        for (int n : SIZES) {
            MemorySegment a = filled(n, 40), b = filled(n, 40);
            oldTensor(a, n).softmaxInPlace(0, n);
            Ops.softmaxInPlace(newView(b, n), 0, n);
            assertBitEqual(a, b, n, "softmax n=" + n);
        }
    }
}
