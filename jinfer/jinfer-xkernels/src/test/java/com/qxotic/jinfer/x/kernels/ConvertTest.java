package com.qxotic.jinfer.x.kernels;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.format.gguf.GGMLType;
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
 * Differential oracles for the span converters: kernels.Convert against the old tensor copy paths
 * on identical inputs. F32→F16 and Q8_0→F32 are deterministic scalar on both sides (bit-equal);
 * F16→F32 uses the same exact vector converter both sides (ulp-bound for tier noise).
 */
class ConvertTest {

    private final Arena arena = Arena.ofAuto();

    private MemoryView<MemorySegment> f16View(MemorySegment seg, long n) {
        return Views.wrap(seg, DataType.FP16, Shape.flat(n));
    }

    private MemoryView<MemorySegment> bf16View(MemorySegment seg, long n) {
        return Views.wrap(seg, DataType.BF16, Shape.flat(n));
    }

    private float getF32(MemorySegment seg, long i) {
        return seg.get(ValueLayout.JAVA_FLOAT_UNALIGNED, 4L * i);
    }

    private short getF16(MemorySegment seg, long i) {
        return seg.get(ValueLayout.JAVA_SHORT_UNALIGNED, 2L * i);
    }

    @Test
    void f16ToF32Parity() {
        for (int n : new int[] {1, 7, 16, 17, 64, 1000}) {
            MemorySegment f16 = arena.allocate(2L * n, 64);
            Random rng = new Random(n);
            for (int i = 0; i < n; i++) {
                f16.set(
                        ValueLayout.JAVA_SHORT_UNALIGNED,
                        2L * i,
                        Float.floatToFloat16(rng.nextFloat() * 4 - 2));
            }
            FloatTensor old = FloatTensor.create(GGMLType.F16, n, f16);
            float[] expected = new float[n];
            old.copyRow(0, expected, 0, n);

            MemorySegment dst = arena.allocate(4L * n, 64);
            Convert.f16ToF32(f16View(f16, n), 0, Oracles.f32View(dst, n), 0, n);
            for (int i = 0; i < n; i++) {
                assertEquals(expected[i], getF32(dst, i), "f16ToF32 n=" + n + " at " + i);
            }
        }
    }

    @Test
    void f32ToF16Parity() {
        for (int n : new int[] {1, 7, 16, 17, 64, 1000}) {
            MemorySegment src = Oracles.f32(arena, n, n);
            MemorySegment dstOld = arena.allocate(2L * n, 64);
            MemorySegment dstNew = arena.allocate(2L * n, 64);
            FloatTensor oldF16 = FloatTensor.create(GGMLType.F16, n, dstOld);
            Oracles.oldF32(src, n).copyTo(0, oldF16, 0, n);
            Convert.f32ToF16(Oracles.f32View(src, n), 0, f16View(dstNew, n), 0, n);
            for (int i = 0; i < n; i++) {
                assertEquals(getF16(dstOld, i), getF16(dstNew, i), "f32ToF16 n=" + n + " at " + i);
            }
        }
    }

    @Test
    void bf16ToF32Parity() {
        for (int n : new int[] {1, 7, 16, 17, 64, 1000}) {
            MemorySegment bf16 = arena.allocate(2L * n, 64);
            Random rng = new Random(n);
            for (int i = 0; i < n; i++)
                bf16.set(
                        ValueLayout.JAVA_SHORT_UNALIGNED,
                        2L * i,
                        com.qxotic.jota.BFloat16.fromFloat(rng.nextFloat() * 4 - 2));
            FloatTensor old = FloatTensor.create(GGMLType.BF16, n, bf16);
            MemorySegment dst = arena.allocate(4L * n, 64);
            Convert.bf16ToF32(bf16View(bf16, n), 0, Oracles.f32View(dst, n), 0, n);
            for (int i = 0; i < n; i++)
                assertEquals(old.getFloat(i), getF32(dst, i), "bf16ToF32 n=" + n + " at " + i);
        }
    }

    @Test
    void dequantQ8_0Parity() {
        int m = 4, k = 2048;
        MemorySegment q8 = Oracles.q8(arena, m, k, 5);
        MemorySegment dstOld = arena.allocate(4L * k, 64);
        MemorySegment dstNew = arena.allocate(4L * k, 64);
        // row 2, plus a mid-block offset case through the row-1 tail
        Oracles.oldQ8(q8, (long) m * k).copyTo((long) 2 * k, Oracles.oldF32(dstOld, k), 0, k);
        Convert.dequantQ8_0(
                Oracles.q8View(q8, (long) m * k), (long) 2 * k, Oracles.f32View(dstNew, k), 0, k);
        for (int i = 0; i < k; i++) {
            assertEquals(getF32(dstOld, i), getF32(dstNew, i), "dequant row at " + i);
        }

        int off = k - 17, count = 40; // crosses a block boundary, unaligned start
        MemorySegment dOld = arena.allocate(4L * count, 64);
        MemorySegment dNew = arena.allocate(4L * count, 64);
        Oracles.oldQ8(q8, (long) m * k).copyTo(off, Oracles.oldF32(dOld, count), 0, count);
        Convert.dequantQ8_0(
                Oracles.q8View(q8, (long) m * k), off, Oracles.f32View(dNew, count), 0, count);
        for (int i = 0; i < count; i++) {
            assertEquals(getF32(dOld, i), getF32(dNew, i), "dequant offset at " + i);
        }
    }

    @Test
    void dequantMxfp4Parity() {
        int n = 96, offset = 17, count = 63;
        MemorySegment weights = Oracles.mxfp4(arena, 1, n, 13);
        MemorySegment expected = arena.allocate(4L * count, 64);
        MemorySegment actual = arena.allocate(4L * count, 64);
        Oracles.oldMxfp4(weights, n).copyTo(offset, Oracles.oldF32(expected, count), 0, count);
        Convert.dequantMxfp4(
                Oracles.mxfp4View(weights, n), offset, Oracles.f32View(actual, count), 0, count);
        for (int i = 0; i < count; i++)
            assertEquals(getF32(expected, i), getF32(actual, i), "MXFP4 at " + i);
    }

    @Test
    void legacyQuantParity() {
        GGMLType[] types = {
            GGMLType.Q4_0,
            GGMLType.Q4_1,
            GGMLType.Q5_1,
            GGMLType.Q4_K,
            GGMLType.Q5_K,
            GGMLType.Q6_K,
            GGMLType.NVFP4,
            GGMLType.Q1_0
        };
        for (GGMLType type : types) {
            long n = type.getElementsPerBlock() * 2L;
            MemorySegment weights = Oracles.legacy(arena, type, n, type.ordinal());
            FloatTensor old = FloatTensor.create(type, n, weights);
            MemoryView<MemorySegment> view = Oracles.legacyView(weights, type, n);
            // whole span, plus an unaligned start crossing block boundaries
            for (long off : new long[] {0, 7}) {
                int count = (int) (n - off);
                MemorySegment expected = arena.allocate(4L * count, 64);
                MemorySegment actual = arena.allocate(4L * count, 64);
                old.copyTo(off, Oracles.oldF32(expected, count), 0, count);
                Convert.copyToF32(view, off, Oracles.f32View(actual, count), 0, count);
                for (int i = 0; i < count; i++) {
                    assertEquals(
                            getF32(expected, i),
                            getF32(actual, i),
                            type + " off=" + off + " at " + i);
                }
            }
        }
    }

    /**
     * copyToF32 routes each dtype to its arm (outputs match the direct-arm calls, which are
     * themselves oracle-tested above) and rejects a dtype with no arm.
     */
    @Test
    void copyToF32Routing() {
        int n = 96; // three Q8_0 blocks, so all three arms get non-trivial spans
        MemorySegment dstArm = arena.allocate(4L * n, 64);
        MemorySegment dstRouted = arena.allocate(4L * n, 64);

        // Q8_0 -> dequantQ8_0
        MemorySegment q8 = Oracles.q8(arena, 1, n, 7);
        Convert.dequantQ8_0(Oracles.q8View(q8, n), 0, Oracles.f32View(dstArm, n), 0, n);
        Convert.copyToF32(Oracles.q8View(q8, n), 0, Oracles.f32View(dstRouted, n), 0, n);
        for (int i = 0; i < n; i++)
            assertEquals(getF32(dstArm, i), getF32(dstRouted, i), "routed Q8_0 at " + i);

        // FP16 -> f16ToF32
        MemorySegment f16 = arena.allocate(2L * n, 64);
        for (int i = 0; i < n; i++)
            f16.set(ValueLayout.JAVA_SHORT_UNALIGNED, 2L * i, Float.floatToFloat16(i * 0.25f - 8));
        Convert.f16ToF32(f16View(f16, n), 0, Oracles.f32View(dstArm, n), 0, n);
        Convert.copyToF32(f16View(f16, n), 0, Oracles.f32View(dstRouted, n), 0, n);
        for (int i = 0; i < n; i++)
            assertEquals(getF32(dstArm, i), getF32(dstRouted, i), "routed FP16 at " + i);

        // BF16 -> bf16ToF32
        MemorySegment bf16 = arena.allocate(2L * n, 64);
        for (int i = 0; i < n; i++)
            bf16.set(
                    ValueLayout.JAVA_SHORT_UNALIGNED,
                    2L * i,
                    com.qxotic.jota.BFloat16.fromFloat(i * 0.25f - 8));
        Convert.bf16ToF32(bf16View(bf16, n), 0, Oracles.f32View(dstArm, n), 0, n);
        Convert.copyToF32(bf16View(bf16, n), 0, Oracles.f32View(dstRouted, n), 0, n);
        for (int i = 0; i < n; i++)
            assertEquals(getF32(dstArm, i), getF32(dstRouted, i), "routed BF16 at " + i);

        // FP32 -> copyF32
        MemorySegment f32 = Oracles.f32(arena, n, 11);
        Convert.copyF32(Oracles.f32View(f32, n), 0, Oracles.f32View(dstArm, n), 0, n);
        Convert.copyToF32(Oracles.f32View(f32, n), 0, Oracles.f32View(dstRouted, n), 0, n);
        for (int i = 0; i < n; i++)
            assertEquals(getF32(dstArm, i), getF32(dstRouted, i), "routed FP32 at " + i);

        // no arm: FP64 rejected
        MemorySegment f64 = arena.allocate(8L * n, 64);
        MemoryView<MemorySegment> f64View = Views.wrap(f64, DataType.FP64, Shape.flat(n));
        assertThrows(
                UnsupportedOperationException.class,
                () -> Convert.copyToF32(f64View, 0, Oracles.f32View(dstRouted, n), 0, n));
    }

    /**
     * gatherToF32 == n per-row copyToF32 calls, bit-identical: the vectorized Q8_0 arm (B2I
     * sign-extends, I2F is exact for a byte, one f32 multiply - no tier noise) and the per-row
     * fallback on an FP32 table. Repeats and unordered rows included.
     */
    @Test
    void gatherToF32Parity() {
        int rows = 8, dim = 96; // 3 Q8_0 blocks per row
        int[] tokens = {3, 0, 7, 7, 2, 5};
        long n = (long) rows * dim, out = (long) tokens.length * dim;

        MemorySegment q8 = Oracles.q8(arena, rows, dim, 7);
        MemoryView<MemorySegment> q8Table = Oracles.q8View(q8, n);
        MemorySegment expected = arena.allocate(4L * out, 64);
        MemorySegment actual = arena.allocate(4L * out, 64);
        for (int r = 0; r < tokens.length; r++) {
            Convert.copyToF32(
                    q8Table,
                    (long) tokens[r] * dim,
                    Oracles.f32View(expected, out),
                    (long) r * dim,
                    dim);
        }
        Convert.gatherToF32(
                q8Table, tokens, 0, tokens.length, Oracles.f32View(actual, out), 0, dim);
        for (long i = 0; i < out; i++) {
            assertEquals(getF32(expected, i), getF32(actual, i), "gather Q8_0 at " + i);
        }

        MemorySegment f32 = Oracles.f32(arena, (int) n, 11);
        MemoryView<MemorySegment> f32Table = Oracles.f32View(f32, n);
        Convert.gatherToF32(
                f32Table, tokens, 0, tokens.length, Oracles.f32View(actual, out), 0, dim);
        for (int r = 0; r < tokens.length; r++) {
            for (int i = 0; i < dim; i++) {
                assertEquals(
                        getF32(f32, (long) tokens[r] * dim + i),
                        getF32(actual, (long) r * dim + i),
                        "gather FP32 row " + r + " at " + i);
            }
        }
    }
}
