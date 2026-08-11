package com.qxotic.jinfer.x;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.format.gguf.GGMLType;
import com.qxotic.jinfer.FloatTensor;
import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Random;
import org.junit.jupiter.api.Test;

/**
 * Differential oracles for the span converters: x.Convert against the old tensor copy paths on
 * identical inputs. F32→F16 and Q8_0→F32 are deterministic scalar on both sides (bit-equal);
 * F16→F32 uses the same exact vector converter both sides (ulp-bound for tier noise).
 */
class ConvertTest {

    private final Arena arena = Arena.ofAuto();

    private MemoryView<MemorySegment> f16View(MemorySegment seg, long n) {
        return Views.wrap(seg, DataType.FP16, Shape.flat(n));
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
}
