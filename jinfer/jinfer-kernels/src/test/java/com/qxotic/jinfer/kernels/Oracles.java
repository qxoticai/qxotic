package com.qxotic.jinfer.kernels;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.format.gguf.GGMLType;
import com.qxotic.jinfer.Views;
import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Random;

/** Shared deterministic test data and numeric assertions for the kernels. */
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

    static MemorySegment mxfp4(Arena arena, int m, int k, long seed) {
        Random rng = new Random(seed);
        MemorySegment seg = arena.allocate((long) m * k / 32 * 17, 64);
        for (long block = 0; block < seg.byteSize() / 17; block++) {
            seg.set(ValueLayout.JAVA_BYTE, block * 17, (byte) 127);
            for (int i = 0; i < 16; i++)
                seg.set(ValueLayout.JAVA_BYTE, block * 17 + 1 + i, (byte) rng.nextInt(256));
        }
        return seg;
    }

    /**
     * Block-quant weights: random payload (every byte pattern is a valid encoding), with every f16
     * scale field pinned to a small sane value so random bits can't produce inf/NaN scales. NVFP4's
     * ue4m3 scales decode to a finite float for any byte.
     */
    static MemorySegment blockQuant(Arena arena, GGMLType type, long elements, long seed) {
        Random rng = new Random(seed);
        MemorySegment seg = arena.allocate(type.byteSizeFor(elements), 64);
        for (long i = 0; i < seg.byteSize(); i++) {
            seg.set(ValueLayout.JAVA_BYTE, i, (byte) rng.nextInt(256));
        }
        long bs = type.getBlockByteSize();
        short d = Float.floatToFloat16(0.015625f), dmin = Float.floatToFloat16(0.0078125f);
        for (long b = 0; b < elements / type.getElementsPerBlock(); b++) {
            long bo = b * bs;
            switch (type) {
                case Q4_0, Q1_0 -> seg.set(ValueLayout.JAVA_SHORT_UNALIGNED, bo, d);
                case Q4_1, Q5_1, Q4_K, Q5_K -> {
                    seg.set(ValueLayout.JAVA_SHORT_UNALIGNED, bo, d);
                    seg.set(ValueLayout.JAVA_SHORT_UNALIGNED, bo + 2, dmin);
                }
                case Q6_K -> seg.set(ValueLayout.JAVA_SHORT_UNALIGNED, bo + 208, d);
                default -> {} // NVFP4: ue4m3 is finite for every byte
            }
        }
        return seg;
    }

    static MemoryView<MemorySegment> blockQuantView(
            MemorySegment seg, GGMLType type, long elements) {
        return Views.wrap(
                seg,
                GGMLDataTypes.toDataType(type),
                Shape.flat(elements / type.getElementsPerBlock())); // shape counts BLOCKS
    }

    static MemoryView<MemorySegment> f32View(MemorySegment seg, long n) {
        return Views.wrap(seg, DataType.FP32, Shape.flat(n));
    }

    static MemoryView<MemorySegment> q8View(MemorySegment seg, long n) {
        return Views.wrap(seg, DataType.Q8_0, Shape.flat(n / 32)); // shape counts BLOCKS
    }

    static MemoryView<MemorySegment> mxfp4View(MemorySegment seg, long n) {
        return Views.wrap(seg, DataType.MXFP4, Shape.flat(n / 32));
    }
}
