package com.qxotic.jinfer.kernels;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import com.qxotic.jinfer.Views;
import com.qxotic.jota.BFloat16;
import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryAllocators;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.junit.jupiter.api.Test;

/** Conversion contracts independent of the retired FloatTensor copy paths. */
class ConvertContractTest {

    @Test
    void denseFloatFormatsDecodeKnownValuesAndOffsets() {
        try (Arena arena = Arena.ofConfined()) {
            var memory = MemoryAllocators.ofArena(arena);
            float[] values = {-3.5f, -0.25f, 0, 1, 9.5f, 0.125f, -2, 4};
            var f32 = Views.fromFloatArray(memory, values);
            var f16 =
                    Views.wrap(
                            arena.allocate(values.length * 2L),
                            DataType.FP16,
                            Shape.flat(values.length));
            var decoded = Views.allocateF32(memory, values.length);

            Convert.f32ToF16(f32, 0, f16, 0, values.length);
            Convert.f16ToF32(f16, 1, decoded, 2, values.length - 2);

            float[] actual = Views.toFloatArray(decoded, "decoded");
            for (int i = 0; i < values.length - 2; i++) {
                actual[i + 2] = values[i + 1] - actual[i + 2];
            }
            assertArrayEquals(new float[values.length], actual, 0f);

            var bf16 =
                    Views.wrap(
                            arena.allocate(values.length * 2L),
                            DataType.BF16,
                            Shape.flat(values.length));
            for (int i = 0; i < values.length; i++) {
                bf16.memory()
                        .base()
                        .set(
                                ValueLayout.JAVA_SHORT_UNALIGNED,
                                bf16.byteOffset() + i * 2L,
                                BFloat16.fromFloat(values[i]));
            }
            Convert.bf16ToF32(bf16, 0, decoded, 0, values.length);
            float[] expected = new float[values.length];
            for (int i = 0; i < values.length; i++)
                expected[i] = BFloat16.toFloat(BFloat16.fromFloat(values[i]));
            assertArrayEquals(expected, Views.toFloatArray(decoded, "decoded"), 0f);
        }
    }

    @Test
    void q8DecodeCrossesABlockBoundary() {
        try (Arena arena = Arena.ofConfined()) {
            var memory = MemoryAllocators.ofArena(arena);
            var encoded = arena.allocate(68, 64);
            encoded.set(ValueLayout.JAVA_SHORT_UNALIGNED, 0, Float.floatToFloat16(0.5f));
            encoded.set(ValueLayout.JAVA_SHORT_UNALIGNED, 34, Float.floatToFloat16(0.25f));
            for (int i = 0; i < 32; i++) {
                encoded.set(ValueLayout.JAVA_BYTE, 2L + i, (byte) (i - 16));
                encoded.set(ValueLayout.JAVA_BYTE, 36L + i, (byte) (16 - i));
            }
            var q8 = Views.wrap(encoded, DataType.Q8_0, Shape.flat(2));
            var decoded = Views.allocateF32(memory, 6);

            Convert.dequantQ8_0(q8, 29, decoded, 0, 6);

            assertArrayEquals(
                    new float[] {6.5f, 7f, 7.5f, 4f, 3.75f, 3.5f},
                    Views.toFloatArray(decoded, "decoded"),
                    0f);
        }
    }

    @Test
    void everyHalfDecodesExactly() {
        // all 65536 halves through the vector body and the scalar tail: subnormals, +-Inf and
        // NaN included, bit-identical to Float.float16ToFloat
        int count = 1 << 16;
        try (Arena arena = Arena.ofConfined()) {
            var memory = MemoryAllocators.ofArena(arena);
            MemorySegment raw = arena.allocate(count * 2L);
            for (int i = 0; i < count; i++) raw.set(ValueLayout.JAVA_SHORT, i * 2L, (short) i);
            var f16 = Views.wrap(raw, DataType.FP16, Shape.flat(count));
            var decoded = Views.allocateF32(memory, count);
            Convert.f16ToF32(f16, 0, decoded, 0, count);
            float[] actual = Views.toFloatArray(decoded, "decoded");
            for (int i = 0; i < count; i++) {
                float expected = Float.float16ToFloat((short) i);
                if (Float.isNaN(expected)) { // payload kept, not quieted: still NaN
                    org.junit.jupiter.api.Assertions.assertTrue(
                            Float.isNaN(actual[i]), "half 0x" + Integer.toHexString(i));
                    continue;
                }
                org.junit.jupiter.api.Assertions.assertEquals(
                        Float.floatToRawIntBits(expected),
                        Float.floatToRawIntBits(actual[i]),
                        "half 0x" + Integer.toHexString(i));
            }
        }
    }
}
