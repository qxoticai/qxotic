package com.qxotic.jota.memory;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.DataType;
import com.qxotic.jota.DeviceType;
import com.qxotic.jota.memory.impl.MemoryAllocatorFactory;
import com.qxotic.jota.memory.impl.MemoryFactory;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

class MemoryTest {

    @Test
    void testAllocate() {
        var allocator = MemoryAllocatorFactory.ofPanama();
        DataType dataType = DataType.FP32;
        long totalBytes = dataType.byteSizeFor(3 * 5);
        try (var memory = allocator.allocateMemory(totalBytes)) {
            assertEquals(3 * 5 * dataType.byteSize(), memory.byteSize());
            assertFalse(memory.isReadOnly());
        }
    }

    @Test
    void testPanamaMemoryBase() {
        try (var arena = Arena.ofShared()) {
            MemorySegment memorySegment = arena.allocate(Float.BYTES * 16);
            Memory<MemorySegment> memory = MemoryFactory.ofMemorySegment(memorySegment);
            assertEquals(memorySegment.byteSize(), memory.byteSize());
            assertSame(memorySegment, memory.base());
        }
    }

    static Stream<Arguments> primitiveArrayMemoryProvider() {
        boolean[] booleans = {false, true};
        byte[] bytes = {1, 2};
        short[] shorts = {1, 2};
        int[] ints = {1, 2};
        long[] longs = {1, 2};
        float[] floats = {1, 2};
        double[] doubles = {1, 2};
        return Stream.of(
                Arguments.of(MemoryFactory.ofBooleans(booleans), booleans, Byte.BYTES),
                Arguments.of(MemoryFactory.ofBytes(bytes), bytes, Byte.BYTES),
                Arguments.of(MemoryFactory.ofShorts(shorts), shorts, Short.BYTES),
                Arguments.of(MemoryFactory.ofInts(ints), ints, Integer.BYTES),
                Arguments.of(MemoryFactory.ofLongs(longs), longs, Long.BYTES),
                Arguments.of(MemoryFactory.ofFloats(floats), floats, Float.BYTES),
                Arguments.of(MemoryFactory.ofDoubles(doubles), doubles, Double.BYTES));
    }

    @ParameterizedTest
    @MethodSource("primitiveArrayMemoryProvider")
    void wrapsPrimitiveArrays(Memory<?> memory, Object array, int elementByteSize) {
        assertSame(array, memory.base());
        assertEquals(2L * elementByteSize, memory.byteSize());
        assertEquals(elementByteSize, memory.memoryGranularity());
        assertFalse(memory.isReadOnly());
        assertTrue(memory.device().belongsTo(DeviceType.JAVA));
    }

    @Test
    void booleanMemoryOnlySupportsBooleanValues() {
        Memory<boolean[]> memory = MemoryFactory.ofBooleans(false);
        assertTrue(memory.supportsDataType(DataType.BOOL));
        assertFalse(memory.supportsDataType(DataType.I8));
    }
}
