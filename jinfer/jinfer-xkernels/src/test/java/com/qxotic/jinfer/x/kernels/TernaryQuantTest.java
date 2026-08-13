package com.qxotic.jinfer.x.kernels;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jinfer.x.Views;
import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.junit.jupiter.api.Test;

class TernaryQuantTest {

    private static final int BLOCK = 256;
    private final Arena arena = Arena.ofAuto();

    @Test
    void decodeBothFormatsAcrossPackingBoundaries() {
        int[] values = values(false);
        for (DataType type : new DataType[] {DataType.TQ1_0, DataType.TQ2_0}) {
            MemorySegment encoded = encode(type, values, 0.5f);
            MemorySegment decoded = arena.allocate(BLOCK * Float.BYTES, 64);
            Convert.copyToF32(view(encoded, type, 1), 0, Oracles.f32View(decoded, BLOCK), 0, BLOCK);
            for (int i = 0; i < BLOCK; i++)
                assertEquals(
                        values[i] * 0.5f,
                        decoded.get(ValueLayout.JAVA_FLOAT_UNALIGNED, 4L * i),
                        type + " at " + i);
        }
    }

    @Test
    void matmulMatchesDecodedWeights() {
        int m = 2, n = 2, k = BLOCK;
        for (DataType type : new DataType[] {DataType.TQ1_0, DataType.TQ2_0}) {
            int[] first = values(false), second = values(true);
            MemorySegment weights = arena.allocate(2 * type.byteSize(), 64);
            MemorySegment.copy(encode(type, first, 0.25f), 0, weights, 0, type.byteSize());
            MemorySegment.copy(
                    encode(type, second, 0.25f), 0, weights, type.byteSize(), type.byteSize());
            MemorySegment activations = arena.allocate((long) n * k * Float.BYTES, 64);
            for (int i = 0; i < n * k; i++)
                activations.set(
                        ValueLayout.JAVA_FLOAT_UNALIGNED, 4L * i, (float) Math.sin(i * 0.07));
            MemorySegment actual = arena.allocate((long) n * m * Float.BYTES, 64);

            MatMul.gemm(
                    view(weights, type, m),
                    Oracles.f32View(activations, (long) n * k),
                    k,
                    Oracles.f32View(actual, (long) n * m),
                    m,
                    m,
                    n,
                    k);

            for (int row = 0; row < n; row++) {
                for (int out = 0; out < m; out++) {
                    int[] quant = out == 0 ? first : second;
                    float expected = 0f;
                    for (int i = 0; i < k; i++)
                        expected +=
                                quant[i]
                                        * 0.25f
                                        * activations.get(
                                                ValueLayout.JAVA_FLOAT_UNALIGNED,
                                                4L * (row * k + i));
                    assertEquals(
                            expected,
                            actual.get(ValueLayout.JAVA_FLOAT_UNALIGNED, 4L * (row * m + out)),
                            2e-5f,
                            type + " row=" + row + " out=" + out);
                }
            }
        }
    }

    private static int[] values(boolean reverse) {
        int[] values = new int[BLOCK];
        for (int i = 0; i < BLOCK; i++) values[i] = ((reverse ? BLOCK - 1 - i : i) % 3) - 1;
        return values;
    }

    private MemorySegment encode(DataType type, int[] values, float scale) {
        MemorySegment block = arena.allocate(type.byteSize(), 64);
        if (type == DataType.TQ1_0) {
            packBase3(block, 0, values, 0, 32, 5);
            packBase3(block, 32, values, 160, 16, 5);
            packBase3(block, 48, values, 240, 4, 4);
            block.set(ValueLayout.JAVA_SHORT_UNALIGNED, 52, Float.floatToFloat16(scale));
        } else {
            for (int half = 0; half < 2; half++) {
                for (int lane = 0; lane < 32; lane++) {
                    int packed = 0;
                    for (int group = 0; group < 4; group++)
                        packed |= (values[half * 128 + group * 32 + lane] + 1) << (2 * group);
                    block.set(ValueLayout.JAVA_BYTE, half * 32L + lane, (byte) packed);
                }
            }
            block.set(ValueLayout.JAVA_SHORT_UNALIGNED, 64, Float.floatToFloat16(scale));
        }
        return block;
    }

    private static void packBase3(
            MemorySegment block,
            int byteOffset,
            int[] values,
            int valueOffset,
            int lanes,
            int digits) {
        int[] weights = {81, 27, 9, 3, 1};
        for (int lane = 0; lane < lanes; lane++) {
            int q = 0;
            for (int digit = 0; digit < digits; digit++)
                q += (values[valueOffset + digit * lanes + lane] + 1) * weights[digit];
            block.set(ValueLayout.JAVA_BYTE, byteOffset + lane, (byte) ((q * 256 + 242) / 243));
        }
    }

    private static MemoryView<MemorySegment> view(
            MemorySegment segment, DataType type, long blocks) {
        return Views.wrap(segment, type, Shape.flat(blocks));
    }
}
