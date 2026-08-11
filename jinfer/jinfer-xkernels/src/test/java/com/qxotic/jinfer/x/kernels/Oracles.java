package com.qxotic.jinfer.x;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.format.gguf.GGMLType;
import com.qxotic.jinfer.F32FloatTensor;
import com.qxotic.jinfer.FloatTensor;
import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Random;

/**
 * Shared oracle plumbing for the differential tests.
 *
 * <p>Why not bit-equal: the Vector API is NOT deterministic across compilation tiers — the
 * interpreter evaluates lanewise {@code fma} as separate mul+add (two roundings) while the JIT
 * intrinsifies it to hardware FMA (one rounding), and parallel rows cross tiers arbitrarily. A
 * probe of old-vs-old (FloatTensor.matmul vs direct per-row dot on identical buffers) shows
 * 670/2048 rows differing by 1-2 ulp. So oracles assert an ulp-level bound: far above tier noise
 * (~1e-7 rel for a 2048-lane dot), far below real divergence (wrong block math, aliasing bugs —
 * observed 100%+).
 */
final class Oracles {

    private Oracles() {}

    static final double REL_TOL = 1e-5;
    static final double ABS_TOL = 1e-6;

    /**
     * Dots accumulate per-lane contraction noise (interpreter mul+add vs JIT fma) over k lanes of
     * O(1) products, and cancellation makes the diff large RELATIVE to a small result: bound the
     * absolute diff instead (1e-4 ≈ 2048 lanes × ~1 ulp × O(1) products, still 100x below any real
     * block-math bug).
     */
    static final double DOT_ABS_TOL = 1e-4;

    static void assertClose(MemorySegment expected, MemorySegment actual, int n, String what) {
        assertClose(expected, actual, n, what, ABS_TOL);
    }

    static void assertClose(
            MemorySegment expected, MemorySegment actual, int n, String what, double absTol) {
        for (int i = 0; i < n; i++) {
            float a = expected.get(ValueLayout.JAVA_FLOAT_UNALIGNED, 4L * i);
            float b = actual.get(ValueLayout.JAVA_FLOAT_UNALIGNED, 4L * i);
            double tol = Math.max(absTol, Math.abs(a) * REL_TOL);
            if (!(Math.abs(a - b) <= tol)) { // also catches NaN
                assertEquals(a, b, tol, what + " at lane " + i);
            }
        }
    }

    static MemorySegment f32(Arena arena, int n, long seed) {
        MemorySegment seg = arena.allocate(4L * n, 64);
        Random rng = new Random(seed);
        for (int i = 0; i < n; i++) {
            seg.set(ValueLayout.JAVA_FLOAT_UNALIGNED, 4L * i, rng.nextFloat() * 4 - 2);
        }
        return seg;
    }

    /** Q8_0 weights, m×k elements: constant small f16 scale, random int8 payloads. */
    static MemorySegment q8(Arena arena, int m, int k, long seed) {
        Random rng = new Random(seed);
        MemorySegment seg = arena.allocate((long) m * k / 32 * 34, 64);
        for (long b = 0; b < seg.byteSize() / 34; b++) {
            seg.set(ValueLayout.JAVA_SHORT_UNALIGNED, b * 34, Float.floatToFloat16(0.015625f));
            for (int i = 0; i < 32; i++) {
                seg.set(ValueLayout.JAVA_BYTE, b * 34 + 2 + i, (byte) rng.nextInt(256));
            }
        }
        return seg;
    }

    static F32FloatTensor oldF32(MemorySegment seg, long n) {
        return (F32FloatTensor) FloatTensor.create(GGMLType.F32, n, seg);
    }

    static FloatTensor oldQ8(MemorySegment seg, long n) {
        return FloatTensor.create(GGMLType.Q8_0, n, seg);
    }

    static MemoryView<MemorySegment> f32View(MemorySegment seg, long n) {
        return Views.wrap(seg, DataType.FP32, Shape.flat(n));
    }

    static MemoryView<MemorySegment> q8View(MemorySegment seg, long n) {
        return Views.wrap(seg, DataType.Q8_0, Shape.flat(n / 32)); // shape counts BLOCKS
    }
}
