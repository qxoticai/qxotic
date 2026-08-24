package com.qxotic.jota.memory;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jota.DataType;
import java.util.stream.Stream;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * Regression tests for {@code FloatsMemoryOperations.fillLong}. The operation packs two 32-bit
 * float bit patterns into one {@code long} and writes them as consecutive floats; a past bug masked
 * each half with 24 bits ({@code & 0xFFFFFF}), zeroing the exponent byte and corrupting every float
 * whose high byte was set.
 */
class FloatsFillLongTest extends AbstractMemoryTest {

    private static final int FLOAT_COUNT = 8;

    /** The two 32-bit halves, both with the exponent byte (bits 24-31) set. */
    private static final long TWO_FLOATS =
            (long) Float.floatToIntBits(1.5f) | ((long) Float.floatToIntBits(-2.0f) << 32);

    static Stream<MemoryDomain<?>> f32Domains() {
        return domainsSupportingF32();
    }

    @ParameterizedTest
    @MethodSource("f32Domains")
    <B> void fillLongWritesBothFloatHalves(MemoryDomain<B> domain) {
        try (domain) {
            MemoryAccess<B> memoryAccess = domain.directAccess();
            Assumptions.assumeTrue(memoryAccess != null, "memory access required");

            Memory<B> memory = allocateFloats(domain.memoryAllocator());
            domain.memoryOperations().fillLong(memory, 0, memory.byteSize(), TWO_FLOATS);

            for (int i = 0; i < FLOAT_COUNT; i += 2) {
                long byteOffset = (long) i * DataType.FP32.byteSize();
                assertEquals(
                        1.5f,
                        memoryAccess.readFloat(memory, byteOffset),
                        0.0f,
                        "low half of pair at index " + i);
                assertEquals(
                        -2.0f,
                        memoryAccess.readFloat(memory, byteOffset + DataType.FP32.byteSize()),
                        0.0f,
                        "high half of pair at index " + i);
            }
        }
    }

    @ParameterizedTest
    @MethodSource("f32Domains")
    <B> void fillLongWritesRepeatedPairsAcrossTheBuffer(MemoryDomain<B> domain) {
        try (domain) {
            MemoryAccess<B> memoryAccess = domain.directAccess();
            Assumptions.assumeTrue(memoryAccess != null, "memory access required");

            Memory<B> memory = allocateFloats(domain.memoryAllocator());
            domain.memoryOperations().fillLong(memory, 0, memory.byteSize(), TWO_FLOATS);

            // Every pair in the buffer must carry the same two floats: the loop writes
            // floatsPerLong elements per iteration, so a wrong stride or index math shows here.
            for (int i = 0; i < FLOAT_COUNT; i++) {
                float expected = (i % 2 == 0) ? 1.5f : -2.0f;
                assertEquals(
                        expected,
                        memoryAccess.readFloat(memory, (long) i * DataType.FP32.byteSize()),
                        0.0f,
                        "element " + i);
            }
        }
    }

    @ParameterizedTest
    @MethodSource("f32Domains")
    <B> void fillLongRespectsByteOffsetAndLength(MemoryDomain<B> domain) {
        try (domain) {
            MemoryAccess<B> memoryAccess = domain.directAccess();
            Assumptions.assumeTrue(memoryAccess != null, "memory access required");

            Memory<B> memory = allocateFloats(domain.memoryAllocator());
            // Baseline so untouched elements are distinguishable from garbage.
            domain.memoryOperations().fillFloat(memory, 0, memory.byteSize(), 7.0f);

            // Fill elements 4..7 (byte offset 16, length 16) with two long-pairs;
            // elements 0..3 must keep the baseline.
            domain.memoryOperations().fillLong(memory, 16, 16, TWO_FLOATS);

            for (int i = 0; i < FLOAT_COUNT; i++) {
                long byteOffset = (long) i * DataType.FP32.byteSize();
                float expected = i < 4 ? 7.0f : (i % 2 == 0 ? 1.5f : -2.0f);
                assertEquals(
                        expected, memoryAccess.readFloat(memory, byteOffset), 0.0f, "element " + i);
            }
        }
    }

    private static <B> Memory<B> allocateFloats(MemoryAllocator<B> allocator) {
        Memory<B> memory = allocator.allocateMemory(DataType.FP32.byteSizeFor(FLOAT_COUNT));
        Assumptions.assumeTrue(memory.supportsDataType(DataType.FP32), "FP32 required");
        return memory;
    }
}
